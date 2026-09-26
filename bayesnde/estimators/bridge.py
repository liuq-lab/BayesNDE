"""Bridge-sampling pointwise marginal density estimation for frozen BGM models."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture

try:
    from bayesnde.estimators.pointwise import ArrayLike, BGM_PointwiseDensityEstimator
except ImportError:  
    from density_estimator import ArrayLike, BGM_PointwiseDensityEstimator


tfd = tfp.distributions


@dataclass(frozen=True)
class BridgeStudentTMixtureProposal:
    weights: np.ndarray
    loc: np.ndarray
    scale_tril: np.ndarray
    df: float
    defensive_weight: float
    covariance_floor: float
    scale_multiplier: float
    covariance_mode: str = "fitted_full"
    center_mode: str = "fitted_gmm"
    scoring: str = "multivariate_t"
    requested_scale: Optional[float] = None


class BGM_BridgeDensityEstimator(BGM_PointwiseDensityEstimator):
    PROPOSAL_FAMILY = (
        "defensive full-covariance mixture-of-Student-t plus standard-normal prior"
    )

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
        bridge_tol: float = 1e-5,
        bridge_max_iter: int = 1000,
        fit_fraction: float = 0.5,
        use_neff: bool = True,
        proposal_scale_multiplier: float = 1.0,
        proposal_covariance_mode: str = "fitted_full",
        proposal_scale: Optional[float] = None,
        proposal_center: str = "fitted_gmm",
        proposal_scoring: str = "multivariate_t",
        return_proposal: bool = False,
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
                bridge_tol=bridge_tol,
                bridge_max_iter=bridge_max_iter,
                fit_fraction=fit_fraction,
                use_neff=use_neff,
                proposal_scale_multiplier=proposal_scale_multiplier,
                proposal_covariance_mode=proposal_covariance_mode,
                proposal_scale=proposal_scale,
                proposal_center=proposal_center,
                proposal_scoring=proposal_scoring,
                return_proposal=return_proposal,
            )
            point_results.append(result)

        log_px_repeats = np.stack([r["log_px_repeats"] for r in point_results], axis=0)
        log_px = self._np_logmeanexp(log_px_repeats, axis=1)
        log_px_mean = np.mean(log_px_repeats, axis=1)
        log_px_sd = (
            np.std(log_px_repeats, axis=1, ddof=1)
            if n_repeats > 1
            else np.zeros(n_points, dtype=np.float64)
        )

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
        density_cfg = self._config.get("density_bridge_eval") or self._config.get(
            "density_eval", {}
        )
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
            bridge_tol=float(density_cfg.get("tol", density_cfg.get("bridge_tol", 1e-5))),
            bridge_max_iter=int(
                density_cfg.get("max_iter", density_cfg.get("bridge_max_iter", 1000))
            ),
            fit_fraction=float(density_cfg.get("fit_fraction", 0.5)),
            use_neff=bool(density_cfg.get("use_neff", True)),
            proposal_scale_multiplier=float(density_cfg.get("proposal_scale_multiplier", 1.0)),
            proposal_covariance_mode=str(density_cfg.get("proposal_covariance_mode", "fitted_full")),
            proposal_scale=(None if density_cfg.get("proposal_scale") is None else float(density_cfg["proposal_scale"])),
            proposal_center=str(density_cfg.get("proposal_center", "fitted_gmm")),
            proposal_scoring=str(density_cfg.get("proposal_scoring", "multivariate_t")),
            return_proposal=bool(density_cfg.get("return_proposal", False)),
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
        bridge_tol: float = 1e-5,
        bridge_max_iter: int = 1000,
        fit_fraction: float = 0.5,
        use_neff: bool = True,
        proposal_scale_multiplier: float = 1.0,
        proposal_covariance_mode: str = "fitted_full",
        proposal_scale: Optional[float] = None,
        proposal_center: str = "fitted_gmm",
        proposal_scoring: str = "multivariate_t",
        return_proposal: bool = False,
    ) -> Dict[str, Any]:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if K < 1:
            raise ValueError("K must be positive.")
        if S < 1:
            raise ValueError("S must be positive.")
        if nu <= 0.0:
            raise ValueError("nu must be positive.")
        if n_repeats < 1:
            raise ValueError("n_repeats must be positive.")
        if bridge_tol <= 0.0:
            raise ValueError("bridge_tol must be positive.")
        if bridge_max_iter < 1:
            raise ValueError("bridge_max_iter must be positive.")
        if eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive.")
        if proposal_scale_multiplier <= 0.0:
            raise ValueError("proposal_scale_multiplier must be positive.")
        if proposal_covariance_mode not in {"fitted_full", "fixed_isotropic"}:
            raise ValueError("proposal_covariance_mode must be 'fitted_full' or 'fixed_isotropic'.")
        if proposal_center not in {"fitted_gmm", "hmc_fit_mean"}:
            raise ValueError("proposal_center must be 'fitted_gmm' or 'hmc_fit_mean'.")
        if proposal_scoring not in {"multivariate_t", "product_univariate_t"}:
            raise ValueError("proposal_scoring must be 'multivariate_t' or 'product_univariate_t'.")
        if proposal_covariance_mode == "fixed_isotropic" and (proposal_scale is None or proposal_scale <= 0.0):
            raise ValueError("A positive proposal_scale is required for fixed_isotropic proposals.")

        hmc_cfg = hmc_settings or {}
        hmc = self.sample_posterior(x_clean, obs_mask_flat, hmc_cfg)
        d_fit, d_bridge, split_diag = self._shuffle_and_split_hmc_samples(
            hmc.samples, fit_fraction=fit_fraction
        )

        proposal = self.fit_student_t_mixture(
            d_fit,
            K=K,
            nu=nu,
            epsilon=epsilon,
            covariance_floor=float(hmc_cfg.get("proposal_covariance_floor", 1e-6)),
            scale_multiplier=float(proposal_scale_multiplier),
            covariance_mode=str(proposal_covariance_mode),
            fixed_scale=proposal_scale,
            center_mode=str(proposal_center),
            scoring=str(proposal_scoring),
        )

        x_tf = tf.convert_to_tensor(x_clean, dtype=self.dtype)
        mask_tf = tf.convert_to_tensor(obs_mask_flat, dtype=tf.bool)
        z_bridge = tf.convert_to_tensor(d_bridge, dtype=self.dtype)
        log_pi_bridge, log_q_bridge = self._evaluate_log_pi_and_log_q(
            z_bridge,
            x_tf,
            mask_tf,
            proposal,
            eval_batch_size=eval_batch_size,
        )
        log_ratio_bridge = log_pi_bridge - log_q_bridge
        self._ensure_finite_tensor("held-out bridge log ratios", log_ratio_bridge)

        ess_bridge = self._bridge_effective_sample_size(
            d_bridge=d_bridge,
            fallback_ess=hmc.min_ess,
            use_neff=use_neff,
        )
        s_p, s_q = self._bridge_sample_fractions(ess_bridge=ess_bridge, S=S)

        log_px_repeats: List[float] = []
        log_z_init_repeats: List[float] = []
        bridge_iterations: List[int] = []
        bridge_converged: List[bool] = []
        bridge_abs_delta: List[float] = []
        proposal_is_ess: List[float] = []
        max_log_ratio: List[float] = []

        for _ in range(n_repeats):
            z_prop = self.sample_defensive_proposal(proposal, S)
            log_pi_prop, log_q_prop = self._evaluate_log_pi_and_log_q(
                z_prop,
                x_tf,
                mask_tf,
                proposal,
                eval_batch_size=eval_batch_size,
            )
            log_ratio_prop = log_pi_prop - log_q_prop
            self._ensure_finite_tensor("proposal log ratios", log_ratio_prop)

            log_z_init = tf.reduce_logsumexp(log_ratio_prop) - tf.math.log(
                tf.cast(tf.size(log_ratio_prop), self.dtype)
            )
            self._ensure_finite_tensor("IS log normalizer initialization", log_z_init)

            log_z, n_iter, converged, abs_delta = self._bridge_fixed_point_tf(
                log_ratio_prop,
                log_ratio_bridge,
                tf.cast(s_p, self.dtype),
                tf.cast(s_q, self.dtype),
                log_z_init,
                tf.cast(bridge_tol, self.dtype),
                tf.cast(bridge_max_iter, tf.int32),
            )
            self._ensure_finite_tensor("bridge log normalizer", log_z)

            converged_bool = bool(converged.numpy())
            if not converged_bool:
                warnings.warn(
                    "Bridge sampling failed to converge within max_iter; returning "
                    "the last fixed-point iterate.",
                    RuntimeWarning,
                )

            log_px_repeats.append(float(log_z.numpy()))
            log_z_init_repeats.append(float(log_z_init.numpy()))
            bridge_iterations.append(int(n_iter.numpy()))
            bridge_converged.append(converged_bool)
            bridge_abs_delta.append(float(abs_delta.numpy()))
            proposal_is_ess.append(float(self._importance_ess(log_ratio_prop).numpy()))
            max_log_ratio.append(float(tf.reduce_max(log_ratio_prop).numpy()))

        diagnostics = {
            "estimator": self.__class__.__name__,
            "proposal_family": self.PROPOSAL_FAMILY,
            "posterior_draws_total": int(hmc.samples.shape[0]),
            "posterior_draws_fit": int(d_fit.shape[0]),
            "posterior_draws_bridge": int(d_bridge.shape[0]),
            "proposal_draws_per_repeat": int(S),
            "posterior_draws_split": True,
            "split_random_shuffle": True,
            "fit_fraction": float(fit_fraction),
            "use_neff": bool(use_neff),
            "ess_bridge": float(ess_bridge),
            "s_p": float(s_p),
            "s_q": float(s_q),
            "bridge_tol": float(bridge_tol),
            "bridge_max_iter": int(bridge_max_iter),
            "proposal_scale_multiplier": float(proposal.scale_multiplier),
            "proposal_covariance_mode": proposal.covariance_mode,
            "proposal_center": proposal.center_mode,
            "proposal_scoring": proposal.scoring,
            "proposal_scale": proposal.requested_scale,
            "proposal_sampling": (
                "product_univariate_student_t_independent_chi_square"
                if proposal.scoring == "product_univariate_t"
                else "multivariate_student_t_shared_chi_square"
            ),
            "proposal_scoring_matches_sampling": True,
            "bridge_iterations_repeats": np.asarray(bridge_iterations, dtype=np.int64),
            "bridge_converged_repeats": np.asarray(bridge_converged, dtype=bool),
            "bridge_abs_delta_repeats": np.asarray(bridge_abs_delta, dtype=np.float64),
            "log_z_init_repeats": np.asarray(log_z_init_repeats, dtype=np.float64),
            "proposal_is_ess_repeats": np.asarray(proposal_is_ess, dtype=np.float64),
            "max_log_ratio_repeats": np.asarray(max_log_ratio, dtype=np.float64),
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
            **split_diag,
        }

        output: Dict[str, Any] = {
            "log_px_repeats": np.asarray(log_px_repeats, dtype=np.float64),
            "diagnostics": diagnostics,
        }

        if return_proposal:
            output["proposal"] = {
                "weights": proposal.weights,
                "loc": proposal.loc,
                "scale_tril": proposal.scale_tril,
                "df": proposal.df,
                "defensive_weight": proposal.defensive_weight,
                "covariance_floor": proposal.covariance_floor,
                "scale_multiplier": proposal.scale_multiplier,
                "covariance_mode": proposal.covariance_mode,
                "center_mode": proposal.center_mode,
                "scoring": proposal.scoring,
                "requested_scale": proposal.requested_scale,
            }

        return output

    def _shuffle_and_split_hmc_samples(
        self,
        samples: np.ndarray,
        fit_fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        sample_array = np.asarray(samples, dtype=np.float64)
        if sample_array.ndim != 2:
            raise ValueError("HMC samples must have shape (M, z_dim).")
        if sample_array.shape[0] < 2:
            raise ValueError("At least two HMC samples are required for fit/bridge splitting.")
        if not (0.0 < fit_fraction < 1.0):
            raise ValueError("fit_fraction must be strictly between 0 and 1.")

        n_total = sample_array.shape[0]
        rng = np.random.default_rng(self._next_numpy_seed())
        perm = rng.permutation(n_total)
        shuffled = sample_array[perm]

        n_fit = int(round(n_total * fit_fraction))
        n_fit = min(max(n_fit, 1), n_total - 1)
        d_fit = shuffled[:n_fit]
        d_bridge = shuffled[n_fit:]

        split_diag = {
            "split_total_after_shuffle": int(n_total),
            "split_fit_count": int(d_fit.shape[0]),
            "split_bridge_count": int(d_bridge.shape[0]),
        }
        return d_fit, d_bridge, split_diag

    def fit_student_t_mixture(
        self,
        posterior_samples: np.ndarray,
        K: int,
        nu: float,
        epsilon: float,
        covariance_floor: float = 1e-6,
        scale_multiplier: float = 1.0,
        covariance_mode: str = "fitted_full",
        fixed_scale: Optional[float] = None,
        center_mode: str = "fitted_gmm",
        scoring: str = "multivariate_t",
    ) -> BridgeStudentTMixtureProposal:
        samples = np.asarray(posterior_samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must have shape (M, z_dim).")
        if not np.all(np.isfinite(samples)):
            raise ValueError("posterior_samples contains non-finite values.")
        if covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be positive.")
        if scale_multiplier <= 0.0:
            raise ValueError("scale_multiplier must be positive.")

        n_samples, z_dim = samples.shape
        n_components = 1 if center_mode == "hmc_fit_mean" else min(int(K), n_samples)
        if n_components < 1:
            raise ValueError("At least one posterior sample is required.")

        if center_mode == "hmc_fit_mean":
            weights = np.ones(1, dtype=np.float64)
            loc = np.mean(samples, axis=0, keepdims=True)
            covariances = np.cov(samples, rowvar=False, ddof=1).reshape(1, z_dim, z_dim)
        else:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=float(covariance_floor),
                random_state=self._next_numpy_seed(),
            )
            gmm.fit(samples)
            weights = np.maximum(gmm.weights_.astype(np.float64), 1e-12)
            weights = weights / np.sum(weights)
            loc = gmm.means_.astype(np.float64)
            covariances = gmm.covariances_.astype(np.float64)

        scale_tril = np.empty((n_components, z_dim, z_dim), dtype=np.float64)
        if covariance_mode == "fixed_isotropic":
            if fixed_scale is None or float(fixed_scale) <= 0.0:
                raise ValueError("fixed_scale must be positive for fixed_isotropic proposals.")
            scale_tril[:] = np.eye(z_dim, dtype=np.float64) * float(fixed_scale)
        else:
            for k in range(n_components):
                scale_tril[k] = self._stable_cholesky(covariances[k], covariance_floor) * float(scale_multiplier)

        return BridgeStudentTMixtureProposal(
            weights=weights,
            loc=loc,
            scale_tril=scale_tril,
            df=float(nu),
            defensive_weight=float(epsilon),
            covariance_floor=float(covariance_floor),
            scale_multiplier=float(scale_multiplier),
            covariance_mode=str(covariance_mode),
            center_mode=str(center_mode),
            scoring=str(scoring),
            requested_scale=None if fixed_scale is None else float(fixed_scale),
        )

    def sample_defensive_proposal(
        self,
        proposal: BridgeStudentTMixtureProposal,
        sample_size: int,
    ) -> tf.Tensor:
        weights = tf.convert_to_tensor(proposal.weights, dtype=self.dtype)
        loc = tf.convert_to_tensor(proposal.loc, dtype=self.dtype)
        scale_tril = tf.convert_to_tensor(proposal.scale_tril, dtype=self.dtype)
        df = tf.cast(proposal.df, self.dtype)
        eps = float(proposal.defensive_weight)

        n = int(sample_size)
        z_dim = int(loc.shape[-1])

        component_ids = tf.squeeze(
            tf.random.categorical(tf.math.log(weights)[tf.newaxis, :], n, seed=self._next_seed()),
            axis=0,
        )
        component_loc = tf.gather(loc, component_ids, axis=0)
        component_scale_tril = tf.gather(scale_tril, component_ids, axis=0)

        gaussian = tf.random.normal((n, z_dim), dtype=self.dtype, seed=self._next_seed())
        if proposal.scoring == "product_univariate_t":
            marginal_scale = tf.linalg.diag_part(component_scale_tril)
            chi2 = tfd.Chi2(df=df).sample(sample_shape=(n, z_dim), seed=self._next_seed())
            t_samples = component_loc + marginal_scale * gaussian / tf.sqrt(chi2 / df)
        else:
            gaussian_scaled = tf.linalg.matvec(component_scale_tril, gaussian)
            chi2 = tfd.Chi2(df=df).sample(sample_shape=(n,), seed=self._next_seed())
            t_samples = component_loc + gaussian_scaled / tf.sqrt(chi2[:, None] / df)

        if eps <= 0.0:
            return t_samples
        prior_samples = tf.random.normal((n, z_dim), dtype=self.dtype, seed=self._next_seed())
        if eps >= 1.0:
            return prior_samples

        defensive_mask = tf.random.uniform((n, 1), dtype=self.dtype, seed=self._next_seed()) < eps
        return tf.where(defensive_mask, prior_samples, t_samples)

    def defensive_mixture_log_prob(
        self,
        z: tf.Tensor,
        proposal: BridgeStudentTMixtureProposal,
    ) -> tf.Tensor:
        z = tf.cast(z, self.dtype)
        weights = tf.convert_to_tensor(proposal.weights, dtype=self.dtype)
        loc = tf.convert_to_tensor(proposal.loc, dtype=self.dtype)
        scale_tril = tf.convert_to_tensor(proposal.scale_tril, dtype=self.dtype)
        df = tf.cast(proposal.df, self.dtype)
        eps = float(proposal.defensive_weight)

        if proposal.scoring == "product_univariate_t":
            component_log_prob = self._student_t_product_log_prob(z, loc, scale_tril, df)
        else:
            component_log_prob = self._student_t_full_log_prob(z, loc, scale_tril, df)
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

    def _student_t_full_log_prob(
        self,
        z: tf.Tensor,
        loc: tf.Tensor,
        scale_tril: tf.Tensor,
        df: tf.Tensor,
    ) -> tf.Tensor:
        z_dim_int = tf.shape(z)[-1]
        z_dim = tf.cast(z_dim_int, self.dtype)
        diff = z[:, None, :] - loc[None, :, :]

        eye = tf.eye(z_dim_int, batch_shape=[tf.shape(scale_tril)[0]], dtype=self.dtype)
        precision = tf.linalg.cholesky_solve(scale_tril, eye)
        maha = tf.einsum("nkd,kde,nke->nk", diff, precision, diff)

        diag = tf.linalg.diag_part(scale_tril)
        log_det_half = tf.reduce_sum(
            tf.math.log(tf.maximum(diag, tf.cast(np.finfo(np.float32).tiny, self.dtype))),
            axis=-1,
        )

        half = tf.cast(0.5, self.dtype)
        log_norm = (
            tf.math.lgamma(half * (df + z_dim))
            - tf.math.lgamma(half * df)
            - half * z_dim * tf.math.log(df * tf.cast(math.pi, self.dtype))
            - log_det_half
        )
        return log_norm[None, :] - half * (df + z_dim) * tf.math.log1p(maha / df)

    def _student_t_product_log_prob(
        self,
        z: tf.Tensor,
        loc: tf.Tensor,
        scale_tril: tf.Tensor,
        df: tf.Tensor,
    ) -> tf.Tensor:
        diagonal = tf.linalg.diag_part(scale_tril)
        tiny = tf.cast(np.finfo(np.float32).tiny, self.dtype)
        diagonal = tf.maximum(diagonal, tiny)
        standardized = (z[:, None, :] - loc[None, :, :]) / diagonal[None, :, :]
        half = tf.cast(0.5, self.dtype)
        log_norm = (
            tf.math.lgamma(half * (df + 1.0))
            - tf.math.lgamma(half * df)
            - half * tf.math.log(df * tf.cast(math.pi, self.dtype))
        )
        per_coordinate = log_norm - tf.math.log(diagonal)[None, :, :] - half * (df + 1.0) * tf.math.log1p(tf.square(standardized) / df)
        return tf.reduce_sum(per_coordinate, axis=-1)

    def _evaluate_log_pi_and_log_q(
        self,
        z: tf.Tensor,
        x_tf: tf.Tensor,
        mask_tf: tf.Tensor,
        proposal: BridgeStudentTMixtureProposal,
        eval_batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        z = tf.convert_to_tensor(z, dtype=self.dtype)
        sample_size = int(z.shape[0])
        if sample_size < 1:
            raise ValueError("At least one latent sample is required.")

        log_pi_parts = []
        log_q_parts = []
        for start in range(0, sample_size, int(eval_batch_size)):
            end = min(start + int(eval_batch_size), sample_size)
            z_batch = z[start:end]
            batch_n = tf.shape(z_batch)[0]
            x_batch = tf.repeat(x_tf, repeats=batch_n, axis=0)
            mask_batch = tf.repeat(mask_tf, repeats=batch_n, axis=0)
            log_pi_parts.append(self.log_joint(z_batch, x_batch, mask_batch))
            log_q_parts.append(self.defensive_mixture_log_prob(z_batch, proposal))

        return tf.concat(log_pi_parts, axis=0), tf.concat(log_q_parts, axis=0)

    @tf.function(reduce_retracing=True)
    def _bridge_fixed_point_tf(
        self,
        log_ratio_prop: tf.Tensor,
        log_ratio_bridge: tf.Tensor,
        s_p: tf.Tensor,
        s_q: tf.Tensor,
        log_z_init: tf.Tensor,
        tol: tf.Tensor,
        max_iter: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        dtype = log_ratio_prop.dtype
        log_s_p = tf.math.log(tf.cast(s_p, dtype))
        log_s_q = tf.math.log(tf.cast(s_q, dtype))
        log_s_count = tf.math.log(tf.cast(tf.size(log_ratio_prop), dtype))
        log_m_count = tf.math.log(tf.cast(tf.size(log_ratio_bridge), dtype))

        def one_update(current_log_z: tf.Tensor) -> tf.Tensor:
            z_term_prop = tf.broadcast_to(
                log_s_q + current_log_z, tf.shape(log_ratio_prop)
            )
            denom_prop = tf.reduce_logsumexp(
                tf.stack([log_s_p + log_ratio_prop, z_term_prop], axis=0),
                axis=0,
            )
            log_num = tf.reduce_logsumexp(log_ratio_prop - denom_prop) - log_s_count

            z_term_bridge = tf.broadcast_to(
                log_s_q + current_log_z, tf.shape(log_ratio_bridge)
            )
            denom_bridge = tf.reduce_logsumexp(
                tf.stack([log_s_p + log_ratio_bridge, z_term_bridge], axis=0),
                axis=0,
            )
            log_den = tf.reduce_logsumexp(-denom_bridge) - log_m_count
            return log_num - log_den

        def cond(
            iteration: tf.Tensor,
            current_log_z: tf.Tensor,
            abs_delta: tf.Tensor,
            converged: tf.Tensor,
        ) -> tf.Tensor:
            del current_log_z, abs_delta
            return tf.logical_and(iteration < max_iter, tf.logical_not(converged))

        def body(
            iteration: tf.Tensor,
            current_log_z: tf.Tensor,
            abs_delta: tf.Tensor,
            converged: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            del abs_delta, converged
            next_log_z = one_update(current_log_z)
            next_delta = tf.abs(next_log_z - current_log_z)
            next_converged = next_delta < tol
            return iteration + 1, next_log_z, next_delta, next_converged

        iteration0 = tf.constant(0, dtype=tf.int32)
        abs_delta0 = tf.constant(math.inf, dtype=dtype)
        converged0 = tf.constant(False)
        iterations, log_z, abs_delta, converged = tf.while_loop(
            cond,
            body,
            loop_vars=(iteration0, log_z_init, abs_delta0, converged0),
        )
        return log_z, iterations, converged, abs_delta

    def _bridge_effective_sample_size(
        self,
        d_bridge: np.ndarray,
        fallback_ess: float,
        use_neff: bool,
    ) -> float:
        n_bridge = int(d_bridge.shape[0])
        if not use_neff:
            return float(n_bridge)

        try:
            bridge_chain = np.asarray(d_bridge, dtype=np.float64)[:, None, :]
            ess_per_dim = self._effective_sample_size(bridge_chain)
            ess_bridge = float(np.min(ess_per_dim))
        except Exception as exc:
            warnings.warn(
                f"Held-out posterior ESS failed ({exc}); using HMC fallback ESS.",
                RuntimeWarning,
            )
            ess_bridge = float(fallback_ess)

        if not np.isfinite(ess_bridge) or ess_bridge <= 0.0:
            warnings.warn(
                "Held-out posterior ESS was unavailable; using physical bridge count.",
                RuntimeWarning,
            )
            return float(n_bridge)
        return float(min(max(ess_bridge, 1.0), float(n_bridge)))

    @staticmethod
    def _bridge_sample_fractions(ess_bridge: float, S: int) -> Tuple[float, float]:
        denom = float(ess_bridge) + float(S)
        if denom <= 0.0:
            raise ValueError("Bridge sample-fraction denominator must be positive.")
        return float(ess_bridge / denom), float(S / denom)

    @staticmethod
    def _stable_cholesky(covariance: np.ndarray, covariance_floor: float) -> np.ndarray:
        cov = np.asarray(covariance, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        dim = cov.shape[0]
        eye = np.eye(dim, dtype=np.float64)

        jitter = float(covariance_floor)
        for _ in range(8):
            try:
                return np.linalg.cholesky(cov + jitter * eye)
            except np.linalg.LinAlgError:
                jitter *= 10.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, float(covariance_floor))
        repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
        repaired = 0.5 * (repaired + repaired.T)
        return np.linalg.cholesky(repaired + float(covariance_floor) * eye)

    def _next_numpy_seed(self) -> Optional[int]:
        seed = self._next_seed()
        if seed is None:
            return None
        return int(seed % (2**32 - 1))

    @staticmethod
    def _ensure_finite_tensor(name: str, tensor: tf.Tensor) -> None:
        if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            raise FloatingPointError(f"Non-finite values encountered in {name}.")

__all__ = ["BGM_BridgeDensityEstimator", "BridgeStudentTMixtureProposal"]
