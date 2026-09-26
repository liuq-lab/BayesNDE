"""Pointwise marginal density estimation for BGM models with important sampling."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from sklearn.mixture import GaussianMixture


from bayesgm.models import BGM, MNISTBGM
from bayesgm.models.networks import BaseVariationalNet


tfd = tfp.distributions
tfm = tfp.mcmc


ArrayLike = Union[np.ndarray, tf.Tensor, Sequence[float]]


def load_keras_weights_compat(model: tf.keras.Model, path: Union[str, Path]) -> None:
    path = Path(path).resolve()
    import h5py

    with h5py.File(path, "r") as handle:
        legacy_hdf5 = "layer_names" in handle.attrs
    if not legacy_hdf5 or not path.name.endswith(".weights.h5"):
        model.load_weights(str(path))
        return
    alias = Path(tempfile.gettempdir()) / f"{path.stem}.{os.getpid()}.legacy.h5"
    try:
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(path)
        model.load_weights(str(alias))
    finally:
        if alias.exists() or alias.is_symlink():
            alias.unlink()


@dataclass(frozen=True)
class StudentTMixtureProposal:
    weights: np.ndarray
    loc: np.ndarray
    scale_diag: np.ndarray
    df: float
    defensive_weight: float


@dataclass(frozen=True)
class HMCResult:
    samples: np.ndarray
    acceptance_rate: float
    ess_per_dim: np.ndarray
    min_ess: float
    mean_ess: float
    requested_samples: int
    retained_samples: int
    num_chains: int
    burn_in: int
    step_size: float
    num_leapfrog_steps: int
    attempts: int


class BGM_PointwiseDensityEstimator:
    MODEL_REGISTRY = {
        "BGM": BGM,
        "MNISTBGM": MNISTBGM,
    }

    def __init__(
        self,
        params: Optional[Mapping[str, Any]] = None,
        model: Optional[Any] = None,
        model_class: Union[str, type] = "BGM",
        likelihood: str = "gaussian",
        dtype: tf.dtypes.DType = tf.float32,
        random_seed: Optional[int] = None,
        variance_floor: float = 1e-6,
        fixed_likelihood_variance: Optional[float] = None,
    ) -> None:
        if model is None and params is None:
            raise ValueError("Either `params` or a pre-built frozen `model` must be provided.")

        self.dtype = dtype
        self.random_seed = random_seed
        self.variance_floor = float(variance_floor)
        self.fixed_likelihood_variance = (
            None if fixed_likelihood_variance is None else float(fixed_likelihood_variance)
        )
        if self.fixed_likelihood_variance is not None and self.fixed_likelihood_variance <= 0.0:
            raise ValueError("fixed_likelihood_variance must be positive when provided.")
        self.likelihood = likelihood.lower()
        self._seed_counter = 0
        self._weights_manifest: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}

        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            np.random.seed(random_seed)

        if model is None:
            frozen_params = dict(params or {})
            frozen_params["use_bnn"] = False
            frozen_params["save_model"] = False
            frozen_params["save_res"] = False
            self.params = frozen_params
            cls = self._resolve_model_class(model_class)
            self.model = cls(params=self.params, random_seed=random_seed)
        else:
            self.model = model
            self.params = dict(getattr(model, "params", params or {}))
            if bool(self.params.get("use_bnn", False)):
                raise ValueError(
                    "BGM_PointwiseDensityEstimator requires a deterministic frozen model. "
                    "Instantiate/load BGM with params['use_bnn'] = False."
                )
            self.params["use_bnn"] = False

        self._freeze_model()

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "BGM_PointwiseDensityEstimator":
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        model_setting = config.get("model_setting", {})
        model_config_path = model_setting.get("model_config")
        if not model_config_path:
            raise ValueError("density config must define model_setting.model_config")

        repo_root = Path(__file__).resolve().parents[2]
        model_config_file = cls._resolve_path(model_config_path, config_path.parent, repo_root)
        with open(model_config_file, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}

        params.update(model_setting.get("params_override", {}) or {})
        if "output_dir" in model_setting:
            params["output_dir"] = model_setting["output_dir"]
        params["use_bnn"] = False
        params["save_model"] = False
        params["save_res"] = False

        estimator = cls(
            params=params,
            model_class=model_setting.get("model_class", "BGM"),
            likelihood=config.get("density_eval", {}).get("likelihood", "gaussian"),
            random_seed=config.get("density_eval", {}).get("seed"),
            variance_floor=config.get("density_eval", {}).get("variance_floor", 1e-6),
        )
        estimator._config = config

        generator_weights = model_setting.get("generator_weights")
        encoder_weights = model_setting.get("encoder_weights")
        checkpoint_dir = model_setting.get("checkpoint_dir")

        if generator_weights:
            generator_path = cls._resolve_path(generator_weights, config_path.parent, repo_root)
            encoder_path = (
                cls._resolve_path(encoder_weights, config_path.parent, repo_root)
                if encoder_weights
                else None
            )
            estimator.load_weights(generator_path, encoder_path)
        elif checkpoint_dir:
            ckpt_path = cls._resolve_path(checkpoint_dir, config_path.parent, repo_root)
            estimator.load_weights_from_run(
                ckpt_path,
                epoch=model_setting.get("epoch"),
                egm_iter=model_setting.get("egm_iter"),
                load_encoder=bool(model_setting.get("load_encoder", True)),
            )

        if bool(config.get("run_management", {}).get("save_documents", True)):
            save_dir = config.get("run_management", {}).get("save_dir", "density_outputs")
            save_path = cls._resolve_path(save_dir, config_path.parent, repo_root)
            estimator.save_run_documents(save_path, config_path=config_path)

        return estimator

    @staticmethod
    def _resolve_path(path_value: Union[str, Path], base_dir: Path, repo_root: Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        base_candidate = (base_dir / path).resolve()
        if base_candidate.exists():
            return base_candidate
        return (repo_root / path).resolve()

    @classmethod
    def _resolve_model_class(cls, model_class: Union[str, type]) -> type:
        if isinstance(model_class, str):
            if model_class not in cls.MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model_class={model_class!r}. "
                    f"Expected one of {sorted(cls.MODEL_REGISTRY)}."
                )
            return cls.MODEL_REGISTRY[model_class]
        return model_class

    def _freeze_model(self) -> None:
        for attr in ("g_net", "e_net", "dz_net", "dx_net"):
            net = getattr(self.model, attr, None)
            if net is not None:
                net.trainable = False

    def _next_seed(self) -> Optional[int]:
        if self.random_seed is None:
            return None
        self._seed_counter += 1
        return int(self.random_seed + 1009 * self._seed_counter)

    def _build_networks_for_loading(self) -> None:
        z_dim = int(self.params["z_dim"])
        self.model.g_net(np.zeros((1, z_dim), dtype=np.float32), training=False)

        e_net = getattr(self.model, "e_net", None)
        if e_net is None:
            return

        if self._is_mnist_model():
            e_net(np.zeros((1, 28, 28, 1), dtype=np.float32), training=False)
        else:
            x_dim = int(self.params["x_dim"])
            e_net(np.zeros((1, x_dim), dtype=np.float32), training=False)

    def _is_mnist_model(self) -> bool:
        return self.model.__class__.__name__ == "MNISTBGM" or self.params.get("dataset") == "MNIST"

    def load_weights(
        self,
        generator_weights: Union[str, Path],
        encoder_weights: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        self._build_networks_for_loading()
        generator_path = Path(generator_weights).resolve()
        if not generator_path.exists():
            raise FileNotFoundError(f"Generator weights not found: {generator_path}")
        manifest: Dict[str, Any] = {"generator_weights": str(generator_path)}
        load_keras_weights_compat(self.model.g_net, generator_path)
        if encoder_weights is not None:
            encoder_path = Path(encoder_weights).resolve()
            if not encoder_path.exists():
                raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")
            load_keras_weights_compat(self.model.e_net, encoder_path)
            manifest["encoder_weights"] = str(encoder_path)

        self._weights_manifest.update(manifest)
        self._freeze_model()
        return manifest


    def load_weights_from_run(
        self,
        checkpoint_dir: Union[str, Path],
        epoch: Optional[int] = None,
        egm_iter: Optional[int] = None,
        load_encoder: bool = True,
    ) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_dir).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        if egm_iter is not None:
            epoch = None
            generator_path = checkpoint_path / f"weights_at_egm_init_{egm_iter}_generator.weights.h5"
        elif epoch is None:
            epoch = self._latest_step(checkpoint_path, r"weights_at_(\d+)_generator\.weights\.h5")
            if epoch is not None:
                generator_path = checkpoint_path / f"weights_at_{epoch}_generator.weights.h5"
        else:
            generator_path = checkpoint_path / f"weights_at_{epoch}_generator.weights.h5"
        if egm_iter is None and epoch is None:
            epoch = self._latest_step(
                checkpoint_path,
                r"weights_at_egm_init_(\d+)_generator\.weights\.h5",
            )
            generator_path = checkpoint_path / f"weights_at_egm_init_{epoch}_generator.weights.h5"

        if (epoch is None and egm_iter is None) or not generator_path.exists():
            raise FileNotFoundError(f"No generator weights found in {checkpoint_path}")

        encoder_path = None
        if load_encoder:
            if egm_iter is None:
                egm_iter = self._latest_step(
                    checkpoint_path,
                    r"weights_at_egm_init_(\d+)_encoder\.weights\.h5",
                )
            if egm_iter is not None:
                encoder_path = checkpoint_path / f"weights_at_egm_init_{egm_iter}_encoder.weights.h5"
                if not encoder_path.exists():
                    encoder_path = None

        manifest = self.load_weights(generator_path, encoder_path)
        manifest.update(
            {
                "checkpoint_dir": str(checkpoint_path),
                "epoch": None if epoch is None else int(epoch),
                "egm_iter": None if egm_iter is None else int(egm_iter),
            }
        )
        self._weights_manifest.update(manifest)
        return manifest

    @staticmethod
    def _latest_step(directory: Path, pattern: str) -> Optional[int]:
        regex = re.compile(pattern)
        latest = None
        for path in directory.glob("*.weights.h5"):
            match = regex.match(path.name)
            if not match:
                continue
            step = int(match.group(1))
            latest = step if latest is None else max(latest, step)
        return latest

    def save_run_documents(
        self,
        save_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "model_params_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.params, f, sort_keys=False)

        estimator_settings = {
            "likelihood": self.likelihood,
            "variance_floor": self.variance_floor,
            "fixed_likelihood_variance": self.fixed_likelihood_variance,
        }
        with open(save_path / "estimator_settings.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(estimator_settings, f, sort_keys=False)

        with open(save_path / "weights_manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self._weights_manifest, f, sort_keys=False)

        if self._config:
            with open(save_path / "density_eval_used.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(self._config, f, sort_keys=False)

        if config_path is not None:
            metadata = {
                "config_path": str(Path(config_path).resolve()),
                "forced_use_bnn": False,
                "estimator": self.__class__.__name__,
            }
            with open(save_path / "density_eval_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    def estimate(
        self,
        x_obs: ArrayLike,
        observed_indices: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        K: int = 5,
        S: int = 40000,
        nu: float = 3.0,
        epsilon: float = 0.05,
        n_repeats: int = 5,
        hmc_settings: Optional[Mapping[str, Any]] = None,
        eval_batch_size: int = 4096,
        prior_proposal_only: bool = False,
        return_proposal: bool = False,
        proposal_mode: str = "hmc_gmm",
        proposal_scale: Optional[float] = None,
        proposal_scale_mode: str = "diagonal",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        x_clean, obs_mask, is_single = self._prepare_observations(x_obs, observed_indices)
        n_points = x_clean.shape[0]

        point_results: List[Dict[str, Any]] = []
        for point_idx in range(n_points):
            result = self.estimate_point(
                x_clean[point_idx : point_idx + 1],
                obs_mask[point_idx : point_idx + 1],
                K=K,
                S=S,
                nu=nu,
                epsilon=epsilon,
                n_repeats=n_repeats,
                hmc_settings=hmc_settings,
                eval_batch_size=eval_batch_size,
                prior_proposal_only=prior_proposal_only,
                return_proposal=return_proposal,
                proposal_mode=proposal_mode,
                proposal_scale=proposal_scale,
                proposal_scale_mode=proposal_scale_mode,
            )
            point_results.append(result)

        log_px_repeats = np.stack([r["log_px_repeats"] for r in point_results], axis=0)
        log_px = self._np_logmeanexp(log_px_repeats, axis=1)
        log_px_mean = np.mean(log_px_repeats, axis=1)
        log_px_sd = np.std(log_px_repeats, axis=1, ddof=1) if n_repeats > 1 else np.zeros(n_points)

        output: Dict[str, Any] = {
            "log_px": log_px[0] if is_single else log_px,
            "log_px_repeats": log_px_repeats[0] if is_single else log_px_repeats,
            "log_px_repeat_mean": log_px_mean[0] if is_single else log_px_mean,
            "log_px_repeat_sd": log_px_sd[0] if is_single else log_px_sd,
            "diagnostics": [r["diagnostics"] for r in point_results],
        }

        if return_proposal:
            output["proposal"] = [r["proposal"] for r in point_results]

        if save_path is not None:
            self._save_estimate_output(save_path, output)

        return output

    def estimate_from_config(
        self,
        x_obs: ArrayLike,
        observed_indices: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        density_cfg = self._config.get("density_eval", {})
        hmc_cfg = self._config.get("hmc_settings", {})
        return self.estimate(
            x_obs=x_obs,
            observed_indices=observed_indices,
            K=int(density_cfg.get("K", 5)),
            S=int(density_cfg.get("S", 40000)),
            nu=float(density_cfg.get("nu", 3.0)),
            epsilon=float(density_cfg.get("epsilon", 0.05)),
            n_repeats=int(density_cfg.get("n_repeats", 5)),
            hmc_settings=hmc_cfg,
            eval_batch_size=int(density_cfg.get("eval_batch_size", 4096)),
            prior_proposal_only=bool(density_cfg.get("prior_proposal_only", False)),
            return_proposal=bool(density_cfg.get("return_proposal", False)),
            proposal_mode=str(density_cfg.get("proposal_mode", "hmc_gmm")),
            proposal_scale=(
                None
                if density_cfg.get("proposal_scale") is None
                else float(density_cfg.get("proposal_scale"))
            ),
            proposal_scale_mode=str(density_cfg.get("proposal_scale_mode", "diagonal")),
            save_path=save_path,
        )

    def estimate_point(
        self,
        x_clean: np.ndarray,
        obs_mask_flat: np.ndarray,
        K: int,
        S: int,
        nu: float,
        epsilon: float,
        n_repeats: int,
        hmc_settings: Optional[Mapping[str, Any]] = None,
        eval_batch_size: int = 4096,
        prior_proposal_only: bool = False,
        return_proposal: bool = False,
        proposal_mode: str = "hmc_gmm",
        proposal_scale: Optional[float] = None,
        proposal_scale_mode: str = "diagonal",
    ) -> Dict[str, Any]:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if K < 1:
            raise ValueError("K must be positive.")
        if S < 1:
            raise ValueError("S must be positive.")
        if nu <= 0:
            raise ValueError("nu must be positive.")
        if n_repeats < 1:
            raise ValueError("n_repeats must be positive.")

        proposal_mode = str(proposal_mode).lower()
        proposal_scale_mode = str(proposal_scale_mode).lower()
        valid_proposal_modes = {"hmc_gmm", "hmc_mean", "encoder"}
        if proposal_mode not in valid_proposal_modes:
            raise ValueError(f"proposal_mode must be one of {sorted(valid_proposal_modes)}, got {proposal_mode!r}.")
        if proposal_scale_mode not in {"diagonal", "scalar"}:
            raise ValueError("proposal_scale_mode must be 'diagonal' or 'scalar'.")

        if prior_proposal_only:
            z_dim = int(self.params["z_dim"])
            proposal = StudentTMixtureProposal(
                weights=np.ones((1,), dtype=np.float64),
                loc=np.zeros((1, z_dim), dtype=np.float64),
                scale_diag=np.ones((1, z_dim), dtype=np.float64),
                df=float(nu),
                defensive_weight=1.0,
            )
            hmc = HMCResult(
                samples=np.zeros((0, z_dim), dtype=np.float32),
                acceptance_rate=float("nan"),
                ess_per_dim=np.full(z_dim, np.nan, dtype=np.float64),
                min_ess=float("nan"),
                mean_ess=float("nan"),
                requested_samples=0,
                retained_samples=0,
                num_chains=0,
                burn_in=0,
                step_size=float("nan"),
                num_leapfrog_steps=0,
                attempts=0,
            )
        elif proposal_mode == "encoder":
            z_dim = int(self.params["z_dim"])
            proposal = self.encoder_student_t_proposal(
                x_clean,
                nu=nu,
                epsilon=epsilon,
                scale=proposal_scale,
                covariance_floor=float((hmc_settings or {}).get("proposal_covariance_floor", 1e-6)),
            )
            hmc = HMCResult(
                samples=np.zeros((0, z_dim), dtype=np.float32),
                acceptance_rate=float("nan"),
                ess_per_dim=np.full(z_dim, np.nan, dtype=np.float64),
                min_ess=float("nan"),
                mean_ess=float("nan"),
                requested_samples=0,
                retained_samples=0,
                num_chains=0,
                burn_in=0,
                step_size=float("nan"),
                num_leapfrog_steps=0,
                attempts=0,
            )
        else:
            hmc = self.sample_posterior(x_clean, obs_mask_flat, hmc_settings or {})
            if proposal_mode == "hmc_gmm":
                proposal = self.fit_student_t_mixture(
                    hmc.samples,
                    K=K,
                    nu=nu,
                    epsilon=epsilon,
                    covariance_floor=float((hmc_settings or {}).get("proposal_covariance_floor", 1e-6)),
                )
            elif proposal_mode == "hmc_mean":
                proposal = self.posterior_mean_student_t_proposal(
                    hmc.samples,
                    nu=nu,
                    epsilon=epsilon,
                    scale=proposal_scale,
                    scale_mode=proposal_scale_mode,
                    covariance_floor=float((hmc_settings or {}).get("proposal_covariance_floor", 1e-6)),
                )
            else:
                raise AssertionError("unreachable proposal_mode branch")

        log_px_repeats = []
        is_ess_repeats = []
        max_log_weight_repeats = []

        x_tf = tf.convert_to_tensor(x_clean, dtype=self.dtype)
        mask_tf = tf.convert_to_tensor(obs_mask_flat, dtype=tf.bool)

        for _ in range(n_repeats):
            z_prop = self.sample_defensive_proposal(proposal, S)
            log_w = self._importance_log_weights(
                z_prop,
                x_tf,
                mask_tf,
                proposal,
                eval_batch_size=eval_batch_size,
            )
            log_px = tf.reduce_logsumexp(log_w) - tf.math.log(tf.cast(S, self.dtype))
            log_px_repeats.append(float(log_px.numpy()))
            is_ess_repeats.append(float(self._importance_ess(log_w).numpy()))
            max_log_weight_repeats.append(float(tf.reduce_max(log_w).numpy()))

        diagnostics = {
            "hmc_acceptance_rate": hmc.acceptance_rate,
            "hmc_min_ess": hmc.min_ess,
            "hmc_mean_ess": hmc.mean_ess,
            "hmc_ess_per_dim": hmc.ess_per_dim,
            "hmc_requested_samples": hmc.requested_samples,
            "hmc_retained_samples": hmc.retained_samples,
            "hmc_num_chains": hmc.num_chains,
            "hmc_burn_in": hmc.burn_in,
            "hmc_step_size": hmc.step_size,
            "hmc_num_leapfrog_steps": hmc.num_leapfrog_steps,
            "hmc_attempts": hmc.attempts,
            "prior_proposal_only": bool(prior_proposal_only),
            "proposal_mode": proposal_mode,
            "proposal_scale_mode": proposal_scale_mode,
            "proposal_scale": None if proposal_scale is None else float(proposal_scale),
            "is_ess_repeats": np.asarray(is_ess_repeats, dtype=np.float64),
            "max_log_weight_repeats": np.asarray(max_log_weight_repeats, dtype=np.float64),
        }

        output: Dict[str, Any] = {
            "log_px_repeats": np.asarray(log_px_repeats, dtype=np.float64),
            "diagnostics": diagnostics,
        }

        if return_proposal:
            output["proposal"] = {
                "weights": proposal.weights,
                "loc": proposal.loc,
                "scale_diag": proposal.scale_diag,
                "df": proposal.df,
                "defensive_weight": proposal.defensive_weight,
            }

        return output

    def sample_posterior(
        self,
        x_clean: np.ndarray,
        obs_mask_flat: np.ndarray,
        hmc_settings: Mapping[str, Any],
    ) -> HMCResult:
        requested_m = int(hmc_settings.get("M", 5000))
        burn_in = int(hmc_settings.get("burn_in", 5000))
        step_size = float(hmc_settings.get("step_size", 0.01))
        num_leapfrog_steps = int(hmc_settings.get("num_leapfrog_steps", 10))
        num_chains = int(hmc_settings.get("num_chains", 4))
        target_accept_prob = float(hmc_settings.get("target_accept_prob", 0.75))
        min_effective_samples = float(hmc_settings.get("min_effective_samples", 1000))
        max_retries = int(hmc_settings.get("max_retries", 1))
        retry_multiplier = float(hmc_settings.get("retry_multiplier", 2.0))
        strict_ess = bool(hmc_settings.get("strict_ess", False))
        initial_state = str(hmc_settings.get("initial_state", "encoder"))
        initial_state_scale = float(hmc_settings.get("initial_state_scale", 0.1))

        if requested_m < 1:
            raise ValueError("hmc_settings.M must be positive.")
        if num_chains < 1:
            raise ValueError("hmc_settings.num_chains must be positive.")

        x_tf = tf.convert_to_tensor(x_clean, dtype=self.dtype)
        mask_tf = tf.convert_to_tensor(obs_mask_flat, dtype=tf.bool)

        current_m = requested_m
        attempts = 0
        last_result = None
        while attempts <= max_retries:
            attempts += 1
            num_results = int(math.ceil(current_m / num_chains))
            samples, accepted = self._run_hmc_chain(
                x_tf=x_tf,
                mask_tf=mask_tf,
                num_results=num_results,
                burn_in=burn_in,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                initial_state=initial_state,
                initial_state_scale=initial_state_scale,
            )

            samples_np = samples.numpy()
            flat_samples = samples_np.reshape((-1, samples_np.shape[-1]))
            ess_per_dim = self._effective_sample_size(samples_np)
            min_ess = float(np.min(ess_per_dim))
            mean_ess = float(np.mean(ess_per_dim))
            acceptance_rate = float(tf.reduce_mean(tf.cast(accepted, tf.float32)).numpy())

            retained = flat_samples.shape[0]
            last_result = HMCResult(
                samples=flat_samples,
                acceptance_rate=acceptance_rate,
                ess_per_dim=ess_per_dim,
                min_ess=min_ess,
                mean_ess=mean_ess,
                requested_samples=requested_m,
                retained_samples=retained,
                num_chains=num_chains,
                burn_in=burn_in,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                attempts=attempts,
            )

            if min_effective_samples <= 0 or min_ess >= min_effective_samples:
                return last_result

            if attempts <= max_retries:
                current_m = int(math.ceil(current_m * retry_multiplier))
                burn_in = int(math.ceil(burn_in * retry_multiplier))

        if last_result is None:
            raise RuntimeError("HMC did not produce samples.")

        message = (
            "Posterior HMC effective sample size is below the configured threshold: "
            f"min_ess={last_result.min_ess:.2f}, threshold={min_effective_samples:.2f}. "
            "The fitted proposal may be less representative; increase M/burn_in or tune HMC settings."
        )
        if strict_ess:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning)
        return last_result

    @tf.function(reduce_retracing=True)
    def _run_hmc_chain_compiled(
        self,
        x_tf: tf.Tensor,
        mask_tf: tf.Tensor,
        init: tf.Tensor,
        seed: Optional[tf.Tensor],
        num_results: int,
        burn_in: int,
        step_size: float,
        num_leapfrog_steps: int,
        target_accept_prob: float,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        def target_log_prob_fn(z: tf.Tensor) -> tf.Tensor:
            chain_count = tf.shape(z)[0]
            x_rep = tf.repeat(x_tf, repeats=chain_count, axis=0)
            mask_rep = tf.repeat(mask_tf, repeats=chain_count, axis=0)
            return self.log_joint(z, x_rep, mask_rep)

        hmc_kernel = tfm.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=tf.cast(step_size, self.dtype),
            num_leapfrog_steps=num_leapfrog_steps,
        )
        adaptive_kernel = tfm.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=int(burn_in * 0.8),
            target_accept_prob=tf.cast(target_accept_prob, self.dtype),
        )

        samples, is_accepted = tfm.sample_chain(
            num_results=num_results,
            num_burnin_steps=burn_in,
            current_state=init,
            kernel=adaptive_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=seed,
        )
        return samples, is_accepted

    def _run_hmc_chain(
        self,
        x_tf: tf.Tensor,
        mask_tf: tf.Tensor,
        num_results: int,
        burn_in: int,
        step_size: float,
        num_leapfrog_steps: int,
        num_chains: int,
        target_accept_prob: float,
        initial_state: str,
        initial_state_scale: float,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        z_dim = int(self.params["z_dim"])
        init = self._make_initial_state(x_tf, num_chains, z_dim, initial_state, initial_state_scale)
        seed_value = self._next_seed()
        seed = (
            None
            if seed_value is None
            else tf.convert_to_tensor([seed_value, seed_value + 1], dtype=tf.int32)
        )
        return self._run_hmc_chain_compiled(
            x_tf=x_tf,
            mask_tf=mask_tf,
            init=init,
            seed=seed,
            num_results=int(num_results),
            burn_in=int(burn_in),
            step_size=float(step_size),
            num_leapfrog_steps=int(num_leapfrog_steps),
            target_accept_prob=float(target_accept_prob),
        )

    def _make_initial_state(
        self,
        x_tf: tf.Tensor,
        num_chains: int,
        z_dim: int,
        initial_state: str,
        initial_state_scale: float,
    ) -> tf.Tensor:
        if initial_state in {"encoder", "mixed_encoder_prior"} and getattr(self.model, "e_net", None) is not None:
            try:
                z0 = self.model.e_net(x_tf, training=False)
                z0 = tf.reshape(tf.cast(z0, self.dtype), [1, z_dim])
                encoder_chains = (
                    num_chains if initial_state == "encoder" else (num_chains + 1) // 2
                )
                noise = tf.random.normal(
                    shape=(encoder_chains, z_dim),
                    mean=0.0,
                    stddev=initial_state_scale,
                    dtype=self.dtype,
                    seed=self._next_seed(),
                )
                encoder_init = tf.repeat(z0, repeats=encoder_chains, axis=0) + noise
                if initial_state == "encoder":
                    return encoder_init

                prior_chains = num_chains - encoder_chains
                if prior_chains <= 0:
                    return encoder_init
                prior_init = tf.random.normal(
                    shape=(prior_chains, z_dim),
                    mean=0.0,
                    stddev=1.0,
                    dtype=self.dtype,
                    seed=self._next_seed(),
                )
                return tf.concat([encoder_init, prior_init], axis=0)
            except Exception as exc:  
                warnings.warn(
                    f"Encoder initial state failed ({exc}); falling back to prior initialization.",
                    RuntimeWarning,
                )

        return tf.random.normal(
            shape=(num_chains, z_dim),
            mean=0.0,
            stddev=1.0,
            dtype=self.dtype,
            seed=self._next_seed(),
        )

    def fit_student_t_mixture(
        self,
        posterior_samples: np.ndarray,
        K: int,
        nu: float,
        epsilon: float,
        covariance_floor: float = 1e-6,
    ) -> StudentTMixtureProposal:
        samples = np.asarray(posterior_samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must have shape (M, z_dim).")
        if not np.all(np.isfinite(samples)):
            raise ValueError("posterior_samples contains non-finite values.")

        n_samples, z_dim = samples.shape
        n_components = min(int(K), n_samples)
        if n_components < 1:
            raise ValueError("At least one posterior sample is required.")

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=float(covariance_floor),
            random_state=self.random_seed,
        )
        gmm.fit(samples)

        weights = np.maximum(gmm.weights_.astype(np.float64), 1e-12)
        weights = weights / np.sum(weights)
        loc = gmm.means_.astype(np.float64)
        scale_diag = np.maximum(gmm.covariances_.astype(np.float64), covariance_floor)

        if scale_diag.shape != (n_components, z_dim):
            raise RuntimeError("Unexpected diagonal covariance shape from GaussianMixture.")

        return StudentTMixtureProposal(
            weights=weights,
            loc=loc,
            scale_diag=scale_diag,
            df=float(nu),
            defensive_weight=float(epsilon),
        )

    def posterior_mean_student_t_proposal(
        self,
        posterior_samples: np.ndarray,
        *,
        nu: float,
        epsilon: float,
        scale: Optional[float] = None,
        scale_mode: str = "diagonal",
        covariance_floor: float = 1e-6,
    ) -> StudentTMixtureProposal:
        samples = np.asarray(posterior_samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must have shape (M, z_dim).")
        if not np.all(np.isfinite(samples)):
            raise ValueError("posterior_samples contains non-finite values.")

        loc = np.mean(samples, axis=0, keepdims=True).astype(np.float64)
        scale_mode = str(scale_mode).lower()
        if scale is not None:
            scalar_variance = float(scale) ** 2
            if scalar_variance <= 0.0:
                raise ValueError("proposal scale must be positive when provided.")
            scale_diag = np.full_like(loc, scalar_variance, dtype=np.float64)
        else:
            sample_var = np.var(samples, axis=0, ddof=1 if samples.shape[0] > 1 else 0)
            if scale_mode == "scalar":
                scalar_variance = float(np.mean(sample_var))
                scale_diag = np.full_like(loc, max(scalar_variance, float(covariance_floor)), dtype=np.float64)
            elif scale_mode == "diagonal":
                scale_diag = np.maximum(sample_var[None, :], float(covariance_floor)).astype(np.float64)
            else:
                raise ValueError("scale_mode must be 'diagonal' or 'scalar'.")

        return StudentTMixtureProposal(
            weights=np.ones((1,), dtype=np.float64),
            loc=loc,
            scale_diag=scale_diag,
            df=float(nu),
            defensive_weight=float(epsilon),
        )

    def encoder_student_t_proposal(
        self,
        x_clean: np.ndarray,
        *,
        nu: float,
        epsilon: float,
        scale: Optional[float] = None,
        covariance_floor: float = 1e-6,
    ) -> StudentTMixtureProposal:
        if getattr(self.model, "e_net", None) is None:
            raise ValueError("encoder proposal requires model.e_net.")
        z_dim = int(self.params["z_dim"])
        x_tf = tf.convert_to_tensor(np.asarray(x_clean, dtype=np.float32), dtype=self.dtype)
        z0 = self.model.e_net(x_tf, training=False)
        loc = np.asarray(tf.reshape(tf.cast(z0, self.dtype), [1, z_dim]).numpy(), dtype=np.float64)
        scalar = float(1.0 if scale is None else scale)
        scalar_variance = scalar * scalar
        if scalar_variance <= 0.0:
            raise ValueError("proposal scale must be positive.")
        scale_diag = np.full_like(loc, scalar_variance, dtype=np.float64)
        return StudentTMixtureProposal(
            weights=np.ones((1,), dtype=np.float64),
            loc=loc,
            scale_diag=scale_diag,
            df=float(nu),
            defensive_weight=float(epsilon),
        )

    def sample_defensive_proposal(self, proposal: StudentTMixtureProposal, sample_size: int) -> tf.Tensor:
        weights = tf.convert_to_tensor(proposal.weights, dtype=self.dtype)
        loc = tf.convert_to_tensor(proposal.loc, dtype=self.dtype)
        scale_diag = tf.convert_to_tensor(proposal.scale_diag, dtype=self.dtype)
        df = tf.cast(proposal.df, self.dtype)
        eps = float(proposal.defensive_weight)

        n = int(sample_size)
        z_dim = int(loc.shape[-1])

        component_ids = tf.squeeze(
            tf.random.categorical(tf.math.log(weights)[tf.newaxis, :], n, seed=self._next_seed()),
            axis=0,
        )
        component_loc = tf.gather(loc, component_ids, axis=0)
        component_scale = tf.gather(scale_diag, component_ids, axis=0)
        gaussian = tf.random.normal((n, z_dim), dtype=self.dtype, seed=self._next_seed())
        chi2 = tfd.Chi2(df=df).sample(sample_shape=(n,), seed=self._next_seed())
        t_samples = component_loc + gaussian * tf.sqrt(component_scale) / tf.sqrt(chi2[:, None] / df)

        if eps <= 0.0:
            return t_samples
        prior_samples = tf.random.normal((n, z_dim), dtype=self.dtype, seed=self._next_seed())
        if eps >= 1.0:
            return prior_samples

        defensive_mask = tf.random.uniform((n, 1), dtype=self.dtype, seed=self._next_seed()) < eps
        return tf.where(defensive_mask, prior_samples, t_samples)

    def log_joint(
        self,
        z: tf.Tensor,
        x: tf.Tensor,
        obs_mask_flat: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        z = tf.cast(z, self.dtype)
        x = tf.cast(x, self.dtype)

        if self.likelihood == "gaussian":
            log_lik = self._gaussian_log_likelihood(z, x, obs_mask_flat)
        elif self.likelihood in {"bernoulli", "bernoulli_logits"}:
            log_lik = self._bernoulli_log_likelihood(z, x, obs_mask_flat)
        else:
            raise ValueError(f"Unsupported likelihood={self.likelihood!r}.")

        return log_lik + self._standard_normal_log_prob(z)

    def _gaussian_log_likelihood(
        self,
        z: tf.Tensor,
        x: tf.Tensor,
        obs_mask_flat: Optional[tf.Tensor],
    ) -> tf.Tensor:
        decoder_outputs = self.model.g_net(z, training=False)
        if not isinstance(decoder_outputs, (tuple, list)) or len(decoder_outputs) != 2:
            raise ValueError("Generator must return a (mu, var) tuple.")
        mu, var = decoder_outputs

        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
        mu_flat = tf.reshape(tf.cast(mu, self.dtype), [tf.shape(mu)[0], -1])
        var_flat = tf.reshape(tf.cast(var, self.dtype), [tf.shape(var)[0], -1])
        if self.fixed_likelihood_variance is not None:
            var_flat = tf.ones_like(var_flat) * tf.cast(self.fixed_likelihood_variance, self.dtype)
        var_flat = tf.maximum(var_flat, tf.cast(self.variance_floor, self.dtype))

        log_2pi = tf.cast(math.log(2.0 * math.pi), self.dtype)
        log_prob = -0.5 * (log_2pi + tf.math.log(var_flat) + tf.square(x_flat - mu_flat) / var_flat)

        if obs_mask_flat is not None:
            mask = tf.cast(obs_mask_flat, self.dtype)
            log_prob = log_prob * mask

        return tf.reduce_sum(log_prob, axis=-1)


    def _bernoulli_log_likelihood(
        self,
        z: tf.Tensor,
        x: tf.Tensor,
        obs_mask_flat: Optional[tf.Tensor],
    ) -> tf.Tensor:
        logits, _ = self.model.g_net(z, training=False)
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
        logits_flat = tf.reshape(tf.cast(logits, self.dtype), [tf.shape(logits)[0], -1])
        log_prob = x_flat * logits_flat - tf.nn.softplus(logits_flat)

        if obs_mask_flat is not None:
            mask = tf.cast(obs_mask_flat, self.dtype)
            log_prob = log_prob * mask

        return tf.reduce_sum(log_prob, axis=-1)

    def defensive_mixture_log_prob(
        self,
        z: tf.Tensor,
        proposal: StudentTMixtureProposal,
    ) -> tf.Tensor:
        z = tf.cast(z, self.dtype)
        weights = tf.convert_to_tensor(proposal.weights, dtype=self.dtype)
        loc = tf.convert_to_tensor(proposal.loc, dtype=self.dtype)
        scale_diag = tf.convert_to_tensor(proposal.scale_diag, dtype=self.dtype)
        df = tf.cast(proposal.df, self.dtype)
        eps = float(proposal.defensive_weight)

        component_log_prob = self._student_t_diag_log_prob(z, loc, scale_diag, df)
        log_t_mix = tf.reduce_logsumexp(tf.math.log(weights)[None, :] + component_log_prob, axis=1)

        if eps <= 0.0:
            return log_t_mix

        log_prior = self._standard_normal_log_prob(z)
        if eps >= 1.0:
            return log_prior

        log_terms = tf.stack(
            [
                tf.math.log(tf.cast(1.0 - eps, self.dtype)) + log_t_mix,
                tf.math.log(tf.cast(eps, self.dtype)) + log_prior,
            ],
            axis=0,
        )
        return tf.reduce_logsumexp(log_terms, axis=0)

    def _student_t_diag_log_prob(
        self,
        z: tf.Tensor,
        loc: tf.Tensor,
        scale_diag: tf.Tensor,
        df: tf.Tensor,
    ) -> tf.Tensor:
        z_dim = tf.cast(tf.shape(z)[-1], self.dtype)
        diff = z[:, None, :] - loc[None, :, :]
        maha = tf.reduce_sum(tf.square(diff) / scale_diag[None, :, :], axis=-1)
        log_det = tf.reduce_sum(tf.math.log(scale_diag), axis=-1)

        half = tf.cast(0.5, self.dtype)
        log_norm = (
            tf.math.lgamma(half * (df + z_dim))
            - tf.math.lgamma(half * df)
            - half * (z_dim * tf.math.log(df * tf.cast(math.pi, self.dtype)) + log_det)
        )
        return log_norm[None, :] - half * (df + z_dim) * tf.math.log1p(maha / df)

    def _standard_normal_log_prob(self, z: tf.Tensor) -> tf.Tensor:
        z_dim = tf.cast(tf.shape(z)[-1], self.dtype)
        log_2pi = tf.cast(math.log(2.0 * math.pi), self.dtype)
        return -0.5 * (tf.reduce_sum(tf.square(z), axis=-1) + z_dim * log_2pi)

    def _importance_log_weights(
        self,
        z_prop: tf.Tensor,
        x_tf: tf.Tensor,
        mask_tf: tf.Tensor,
        proposal: StudentTMixtureProposal,
        eval_batch_size: int,
    ) -> tf.Tensor:
        parts = []
        sample_size = int(z_prop.shape[0])
        for start in range(0, sample_size, int(eval_batch_size)):
            end = min(start + int(eval_batch_size), sample_size)
            z_batch = z_prop[start:end]
            batch_n = tf.shape(z_batch)[0]
            x_batch = tf.repeat(x_tf, repeats=batch_n, axis=0)
            mask_batch = tf.repeat(mask_tf, repeats=batch_n, axis=0)
            log_pi = self.log_joint(z_batch, x_batch, mask_batch)
            log_q = self.defensive_mixture_log_prob(z_batch, proposal)
            parts.append(log_pi - log_q)
        return tf.concat(parts, axis=0)

    def _importance_ess(self, log_w: tf.Tensor) -> tf.Tensor:
        return tf.exp(2.0 * tf.reduce_logsumexp(log_w) - tf.reduce_logsumexp(2.0 * log_w))

    def _effective_sample_size(self, samples: np.ndarray) -> np.ndarray:
        try:
            ess = tfm.effective_sample_size(tf.convert_to_tensor(samples, dtype=self.dtype)).numpy()
            ess = np.asarray(ess, dtype=np.float64)
            if ess.ndim == 2:
                ess = np.sum(ess, axis=0)
            elif ess.ndim > 2:
                ess = np.sum(ess.reshape((-1, ess.shape[-1])), axis=0)
            return np.maximum(ess, 1.0)
        except Exception as exc:
            warnings.warn(
                f"TFP effective_sample_size failed ({exc}); using retained sample count.",
                RuntimeWarning,
            )
            return np.full(samples.shape[-1], samples.reshape((-1, samples.shape[-1])).shape[0], dtype=np.float64)

    def _prepare_observations(
        self,
        x_obs: ArrayLike,
        observed_indices: Optional[Union[Sequence[int], Sequence[Sequence[int]]]],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        x = np.asarray(x_obs, dtype=np.float32)

        if x.ndim == 0:
            raise ValueError("x_obs must contain at least one observed dimension.")

        if x.ndim == 1 or (self._is_mnist_model() and x.ndim == 3):
            x = x[None, ...]
            is_single = True
        else:
            is_single = False

        flat_dim = int(np.prod(x.shape[1:]))
        finite_mask = ~np.isnan(x).reshape((x.shape[0], flat_dim))
        obs_mask = finite_mask.copy()

        if observed_indices is not None:
            explicit_mask = np.zeros_like(obs_mask, dtype=bool)
            if self._is_list_of_lists(observed_indices):
                if len(observed_indices) != x.shape[0]:
                    raise ValueError("Per-point observed_indices length must match number of observations.")
                for i, row in enumerate(observed_indices):
                    explicit_mask[i, np.asarray(row, dtype=np.int64)] = True
            else:
                idx = np.asarray(observed_indices, dtype=np.int64)
                if idx.ndim == 1:
                    explicit_mask[:, idx] = True
                elif idx.ndim == 2:
                    if idx.shape[0] != x.shape[0]:
                        raise ValueError("2-D observed_indices must have one row per observation.")
                    for i, row in enumerate(idx):
                        explicit_mask[i, row] = True
                else:
                    raise ValueError("observed_indices must be a 1-D shared index set or per-point 2-D indices.")
            obs_mask &= explicit_mask

        if not np.all(np.any(obs_mask, axis=1)):
            raise ValueError("Each point must have at least one observed feature.")

        x_clean = np.nan_to_num(x, nan=0.0).astype(np.float32, copy=False)
        return x_clean, obs_mask, is_single

    @staticmethod
    def _is_list_of_lists(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and isinstance(value[0], (list, tuple, np.ndarray))
        )

    @staticmethod
    def _np_logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
        max_value = np.max(values, axis=axis, keepdims=True)
        return (
            np.log(np.mean(np.exp(values - max_value), axis=axis))
            + np.squeeze(max_value, axis=axis)
        )

    @staticmethod
    def _save_estimate_output(save_path: Union[str, Path], output: Mapping[str, Any]) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            log_px=output["log_px"],
            log_px_repeats=output["log_px_repeats"],
            log_px_repeat_mean=output["log_px_repeat_mean"],
            log_px_repeat_sd=output["log_px_repeat_sd"],
            diagnostics=np.asarray(output["diagnostics"], dtype=object),
        )


__all__ = ["BGM_PointwiseDensityEstimator", "StudentTMixtureProposal", "HMCResult"]
