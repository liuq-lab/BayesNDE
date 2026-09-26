#!/usr/bin/env python3
"""Configuration and reproduction entry point for the UCI experiments."""
from __future__ import annotations


import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from experiments.uci import core, data
from experiments.uci.core import (
    LEGACY_PROPOSAL_CONVENTION,
    HERE,
    install_legacy_proposal,
    isolate_base,
    load_shared,
)

REPRODUCTION_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "reproduction.yaml"


DATASET = data.PARKTELE


runner = load_shared("parktele_bgmbs_runner", "protocol_unconditional.py")


isolate_base(runner)


runner.DATASET = DATASET


runner.X_DIM = data.PARKTELE_X_DIM


runner.Z_DIMS = (8,)


runner.DEFAULT_TEST_EPSILON = 0.0


runner.PROPOSAL_MODES = ("legacy_product_q",)


runner.TEST_SHARDS = 6


runner.TOP_K_PER_Z = 3


runner.OUTPUT_ROOT = HERE / "outputs" / DATASET / "bgmbs-full-v1"


runner.base.load_split = data.load_split


SELECTION_CRITERION = "argmax validation decoder log-likelihood (validation only)"


SPECS: dict[str, dict[str, Any]] = {
    "Pendigits10": {
        "round": "full",
        "run_name": "conditional-bgmbs-full-v1",
        "x_dim": 16,
        "y_dim": 10,
        "network_type": "residual",
        "z_epsilon": ((8, 0.0),),
        "test_points": None,
        "test_shards": 8,
        "source_network": "outputs/indep_gmm_bgmbs_highdim_egm_selected_v1",
    },
    "EEGEye": {
        "round": "decoderll",
        "run_name": "conditional-bgmbs-decoderll-v1",
        "x_dim": 14,
        "y_dim": 2,
        "network_type": "residual",
        "z_epsilon": ((5, 0.0),),
        "test_points": None,
        "test_shards": 12,
        "source_network": "outputs/indep_gmm_bgmbs_highdim_egm_selected_v1",
    },
    "Vehicle": {
        "round": "decoderll",
        "run_name": "conditional-bgmbs-decoderll-v1",
        "x_dim": 18,
        "y_dim": 4,
        "network_type": "residual",
        "z_epsilon": ((6, 0.0),),
        "test_points": None,
        "test_shards": 4,
        "source_network": "outputs/indep_gmm_bgmbs_highdim_egm_selected_v1",
    },
}


ALL_DATASETS = tuple(SPECS)


ROUNDS = ("full", "decoderll")


ALIASES = {
    "pendigits": "Pendigits10", "pendigits10": "Pendigits10",
    "eegeye": "EEGEye", "eeg": "EEGEye",
    "vehicle": "Vehicle",
}


RUNNER_ALIAS = {
    "full": "expansion_conditional_runner",
    "decoderll": "cond2_conditional_runner",
}


PROPOSAL_MODE = "legacy"


def canonical_dataset(value: str) -> str:
    try:
        return ALIASES[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"dataset must be one of {sorted(SPECS)}, got {value!r}") from exc


def dataset_round(dataset: str) -> str:
    return SPECS[canonical_dataset(dataset)]["round"]


def architecture_rule(x_dim: int) -> str:
    return "mlp" if int(x_dim) <= 10 else "residual"


def base_output_root(dataset: str) -> Path:
    dataset = canonical_dataset(dataset)
    return HERE / "outputs" / dataset / SPECS[dataset]["run_name"]


def output_root(dataset: str) -> Path:
    dataset = canonical_dataset(dataset)
    root = base_output_root(dataset)
    if dataset_round(dataset) == "decoderll":
        return root.with_name(root.name + "-legacyq")
    return root


def load_split(dataset: str):
    dataset = canonical_dataset(dataset)
    split = data.load_split(dataset)
    spec = SPECS[dataset]
    if split.x_dim != int(spec["x_dim"]) or split.y_dim != int(spec["y_dim"]):
        raise ValueError(
            f"{dataset}: x_dim={split.x_dim} y_dim={split.y_dim}, "
            f"expected {spec['x_dim']}/{spec['y_dim']}"
        )
    return split


_RUNNERS: dict[str, ModuleType] = {}


_PROPOSAL_PATCH: dict[str, Any] = {}


def get_runner(dataset: str) -> ModuleType:
    global _PROPOSAL_PATCH

    dataset = canonical_dataset(dataset)
    which = dataset_round(dataset)
    if which in _RUNNERS:
        return _RUNNERS[which]

    if which == "decoderll":
        _PROPOSAL_PATCH = install_legacy_proposal()

    runner = load_shared(RUNNER_ALIAS[which], "protocol_conditional.py")
    runner.DATASET_SPECS = {**runner.DATASET_SPECS, **SPECS}
    own = tuple(d for d, s in SPECS.items() if s["round"] == which)

    def _canonical(value: str, _own: tuple[str, ...] = own) -> str:
        name = canonical_dataset(value)
        if name not in _own:
            raise ValueError(f"dataset must be one of {sorted(_own)}, got {value!r}")
        return name

    runner.canonical_dataset = _canonical
    runner.load_split = load_split
    runner.output_root = output_root
    runner.model_params = _model_params_for(runner)
    if which == "decoderll":
        runner.TOP_K = 1
        runner.select_iterative = _select_iterative_for(runner)
    _RUNNERS[which] = runner
    return runner


def _model_params_for(runner: ModuleType):
    shared = runner.model_params

    def model_params(dataset: str, z_dim: int, out_dir: Path) -> dict[str, Any]:
        params = shared(dataset, z_dim, out_dir)
        params["y_dim"] = int(runner.DATASET_SPECS[dataset]["y_dim"])
        params["x_cycle_weight"] = 3.0
        return params

    return model_params


def decoder_log_likelihood(model, val_x: np.ndarray, val_y_onehot: np.ndarray,
                           batch_size: int = 1024) -> float:
    import tensorflow as tf

    chunks = []
    for start in range(0, len(val_x), batch_size):
        x = tf.convert_to_tensor(val_x[start:start + batch_size], tf.float32)
        y = tf.convert_to_tensor(val_y_onehot[start:start + batch_size], tf.float32)
        z = model.encode(x, y, training=False)
        mean, variance, _ = model._decode_generator(z, y, training=False)
        variance = tf.maximum(variance, tf.cast(1.0e-6, variance.dtype))
        logp = -0.5 * tf.reduce_sum(
            tf.math.log(tf.cast(2.0 * math.pi, variance.dtype) * variance)
            + tf.square(x - mean) / variance,
            axis=1,
        )
        chunks.append(np.asarray(logp.numpy(), dtype=np.float64))
    return float(np.mean(np.concatenate(chunks)))


def _select_iterative_for(runner: ModuleType):
    def select_iterative(dataset: str) -> dict[str, Any]:
        import yaml
        from bayesnde.estimators.conditional import one_hot

        dataset = canonical_dataset(dataset)
        frozen = output_root(dataset) / "frozen_iterative_top3.json"
        if frozen.exists():
            return json.loads(frozen.read_text())
        shared = base_output_root(dataset) / "frozen_iterative_top3.json"
        if shared.exists():
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(shared, frozen)
            return json.loads(frozen.read_text())

        split = load_split(dataset)
        val_x = np.asarray(split.val_x, dtype=np.float32)
        val_y = one_hot(np.asarray(split.val_y), split.y_dim)

        selected: dict[str, list[dict[str, Any]]] = {}
        scored_all: dict[str, list[dict[str, Any]]] = {}
        for z_dim, epsilon in runner.DATASET_SPECS[dataset]["z_epsilon"]:
            rows: list[dict[str, Any]] = []
            model = None
            for variant in runner.ITERATIVE_VARIANTS:
                root = runner.iterative_root(dataset, int(z_dim), variant)
                if not (root / "complete.json").exists():
                    raise FileNotFoundError(f"Incomplete iterative run: {root}")
                for row in json.loads((root / "generation_curve.json").read_text())["rows"]:
                    config = yaml.safe_load(Path(str(row["config_path"])).read_text())
                    if model is None:
                        model = runner.build_model(config)
                        model.e_net.load_weights(str(row["encoder_weights"]))
                    model.g_net.load_weights(str(row["generator_weights"]))
                    rows.append({**row,
                                 "validation_decoder_mean_log_likelihood":
                                     decoder_log_likelihood(model, val_x, val_y)})
            finite = [r for r in rows
                      if np.isfinite(r["validation_decoder_mean_log_likelihood"])]
            if not finite:
                raise RuntimeError(f"{dataset} z={z_dim}: no finite decoder log-likelihood")
            best = max(finite, key=lambda r: r["validation_decoder_mean_log_likelihood"])
            selected[str(int(z_dim))] = [
                {**best, "rank_within_setting": 1, "rank_within_z_dim": 1}
            ]
            scored_all[str(int(z_dim))] = [
                {k: r[k] for k in ("variant", "epoch",
                                   "validation_decoder_mean_log_likelihood")} for r in rows
            ]

        payload = {
            "dataset": dataset,
            "criterion": SELECTION_CRITERION,
            "validation_points": int(len(val_x)),
            "selected_top3_by_z_dim": selected,
            "all_scored_by_z_dim": scored_all,
            "test_points_used_for_selection": 0,
            "frozen_before_test": True,
            "proposal_mode": PROPOSAL_MODE,
            "proposal_patch": _PROPOSAL_PATCH,
        }
        runner.atomic_json(frozen, payload)
        return payload

    return select_iterative


UNCONDITIONAL_NEW = ("ParkinsonsTelemonitoring",)


UNCONDITIONAL_LEGACY = ("BANK",)


UNCONDITIONAL = UNCONDITIONAL_NEW + UNCONDITIONAL_LEGACY


CONDITIONAL_NEW = ("Pendigits10",)


CONDITIONAL_LEGACY: tuple[str, ...] = ()


CONDITIONAL = CONDITIONAL_NEW + CONDITIONAL_LEGACY


DECODERLL_DATASETS = UNCONDITIONAL + CONDITIONAL


CRITERION = "argmax validation_decoder_mean_log_likelihood (validation only)"


PREREGISTRATION = HERE / "outputs" / "_summary" / "preregistered_secondary_selection.json"


def primary_root(dataset: str) -> Path:
    if dataset in UNCONDITIONAL_LEGACY:
        return HERE / "outputs" / dataset / "bgm-bs-tuning-v2"
    if dataset in UNCONDITIONAL:
        return HERE / "outputs" / dataset / "bgmbs-full-v1"
    if dataset in CONDITIONAL_LEGACY:
        return HERE / "outputs" / dataset / "conditional-bgmbs-tuning-v1"
    return HERE / "outputs" / dataset / "conditional-bgmbs-full-v1"


def secondary_root(dataset: str) -> Path:
    if dataset in UNCONDITIONAL:
        return HERE / "outputs" / dataset / "bgmbs-secondary-v1"
    return HERE / "outputs" / dataset / "conditional-bgmbs-secondary-v1"


def _uncond_runner(dataset: str = "ParkinsonsTelemonitoring"):
    if dataset in UNCONDITIONAL_LEGACY:
        inner = load_shared("legacyq_bank_runner", "protocol_unconditional.py")
        isolate_base(inner)
        inner.PROPOSAL_MODES = ("legacy_product_q",)
    else:
        inner = runner
    runner_ = inner
    runner_.OUTPUT_ROOT = secondary_root(dataset)
    runner_.TOP_K_PER_Z = 1
    return runner_


def _cond_runner(dataset: str):
    inner = get_runner(dataset)
    inner.TOP_K = 1
    root = secondary_root(dataset)
    inner.output_root = lambda ds, _root=root: _root
    return inner


def score(dataset: str) -> dict[str, Any]:
    out = secondary_root(dataset) / "secondary_decoder_ll.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))

    if dataset in UNCONDITIONAL:
        runner = _uncond_runner(dataset)
        rows_by_z: dict[str, list[dict[str, Any]]] = {}
        for z_dim in runner.Z_DIMS:
            collected = []
            for variant in runner.ITERATIVE_VARIANTS:
                path = (
                    primary_root(dataset) / "iterative" / f"zdim_{z_dim}" / variant
                    / "generation_curve.json"
                )
                for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
                    collected.append(row)
            rows_by_z[str(z_dim)] = collected
        payload = {
            "dataset": dataset,
            "criterion": CRITERION,
            "source": "validation_decoder_mean_log_likelihood already recorded by the iterative stage",
            "rows_by_setting": rows_by_z,
        }
    else:
        payload = _score_conditional(dataset)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _score_conditional(dataset: str) -> dict[str, Any]:
    import tensorflow as tf
    from bayesnde.estimators.conditional import one_hot

    runner = _cond_runner(dataset)
    split = runner.load_split(dataset)
    val_x = np.asarray(split.val_x, dtype=np.float32)
    val_y = one_hot(np.asarray(split.val_y), split.y_dim)

    def decoder_ll(model, batch_size: int = 1024) -> float:
        chunks = []
        for start in range(0, len(val_x), batch_size):
            x = tf.convert_to_tensor(val_x[start : start + batch_size], tf.float32)
            y = tf.convert_to_tensor(val_y[start : start + batch_size], tf.float32)
            z = model.encode(x, y, training=False)
            mean, variance, _ = model._decode_generator(z, y, training=False)
            variance = tf.maximum(variance, tf.cast(1.0e-6, variance.dtype))
            logp = -0.5 * tf.reduce_sum(
                tf.math.log(tf.cast(2.0 * math.pi, variance.dtype) * variance)
                + tf.square(x - mean) / variance,
                axis=1,
            )
            chunks.append(np.asarray(logp.numpy(), dtype=np.float64))
        return float(np.mean(np.concatenate(chunks)))

    rows_by_setting: dict[str, list[dict[str, Any]]] = {}
    spec = runner.DATASET_SPECS[dataset]
    for z_dim, epsilon in spec["z_epsilon"]:
        collected: list[dict[str, Any]] = []
        model = None
        for variant in runner.ITERATIVE_VARIANTS:
            path = (
                primary_root(dataset) / "iterative" / f"zdim_{int(z_dim)}" / variant
                / "generation_curve.json"
            )
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            for row in rows:
                config = runner.yaml.safe_load(
                    Path(str(row["config_path"])).read_text(encoding="utf-8")
                )
                if model is None:
                    model = runner.build_model(config)
                    model.e_net.load_weights(str(row["encoder_weights"]))
                model.g_net.load_weights(str(row["generator_weights"]))
                collected.append(
                    {**row, "validation_decoder_mean_log_likelihood": decoder_ll(model)}
                )
        rows_by_setting[str(int(z_dim))] = collected
    return {
        "dataset": dataset,
        "criterion": CRITERION,
        "source": "recomputed with the same estimator the unconditional stage records",
        "validation_points": int(len(val_x)),
        "rows_by_setting": rows_by_setting,
    }


def _primary_rank_context(
    dataset: str, rows: list[dict[str, Any]], best: dict[str, Any]
) -> dict[str, Any]:
    try:
        if dataset in UNCONDITIONAL:
            runner = _uncond_runner(dataset)
            ranked = sorted(
                runner.add_rank_scores(rows),
                key=lambda r: (r["generation_rank_sum"], r["epoch"]),
            )
        else:
            runner = _cond_runner(dataset)
            ranked = runner.rank_rows(rows)
    except Exception:
        return {"generation_rank_sum": float("nan"), "primary_rank_position": None}
    key = (best.get("variant"), int(best.get("epoch", -1)))
    for position, row in enumerate(ranked, start=1):
        if (row.get("variant"), int(row.get("epoch", -2))) == key:
            return {
                "generation_rank_sum": float(row.get("generation_rank_sum", float("nan"))),
                "primary_rank_position": position,
                "primary_rank_pool_size": len(ranked),
            }
    return {"generation_rank_sum": float("nan"), "primary_rank_position": None}


def select(dataset: str) -> dict[str, Any]:
    root = secondary_root(dataset)
    frozen = root / "frozen_iterative_top3.json"
    if frozen.exists():
        return json.loads(frozen.read_text(encoding="utf-8"))

    scored = score(dataset)
    selected: dict[str, list[dict[str, Any]]] = {}
    for setting, rows in scored["rows_by_setting"].items():
        finite = [
            r for r in rows
            if np.isfinite(r.get("validation_decoder_mean_log_likelihood", np.nan))
        ]
        if not finite:
            raise RuntimeError(f"{dataset} setting {setting}: no finite decoder log-likelihood")
        best = max(finite, key=lambda r: r["validation_decoder_mean_log_likelihood"])
        extra = _primary_rank_context(dataset, rows, best)
        selected[setting] = [
            {**best, **extra, "rank_within_z_dim": 1, "rank_within_setting": 1}
        ]

    payload = {
        "dataset": dataset,
        "criterion": CRITERION,
        "arm": "secondary",
        "preregistration": str(PREREGISTRATION),
        "primary_run_root": str(primary_root(dataset)),
        "selected_top3_by_z_dim": selected,
        "test_points_used_for_selection": 0,
        "frozen_before_test": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _runner_for(dataset: str):
    if dataset in UNCONDITIONAL:
        return _uncond_runner(dataset)
    return _cond_runner(dataset)


def decoderll_task_count(dataset: str) -> int:
    runner = _runner_for(dataset)
    if dataset in UNCONDITIONAL:
        return len(runner.Z_DIMS) * 1 * len(runner.PROPOSAL_MODES) * int(runner.TEST_SHARDS)
    spec = runner.DATASET_SPECS[dataset]
    return len(spec["z_epsilon"]) * 1 * int(spec["test_shards"])


def decoderll_test_shard(dataset: str, task: int) -> dict[str, Any]:
    select(dataset)
    runner = _runner_for(dataset)
    if dataset in UNCONDITIONAL:
        return runner.run_test_shard(task)
    return runner.run_test_shard(dataset, task)


def decoderll_merge_test(dataset: str) -> dict[str, Any]:
    select(dataset)
    runner = _runner_for(dataset)
    return runner.merge_test() if dataset in UNCONDITIONAL else runner.merge_test(dataset)


def decoderll_validate(dataset: str) -> dict[str, Any]:
    frozen = secondary_root(dataset) / "frozen_iterative_top3.json"
    if not frozen.exists():
        return {
            "dataset": dataset,
            "arm": "decoderll",
            "criterion": CRITERION,
            "selection_frozen": False,
            "note": "run `score` first; `select` then freezes the candidates",
            "secondary_root": str(secondary_root(dataset)),
            "test_jobs": decoderll_task_count(dataset),
        }
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    rows = {
        setting: {
            "variant": cand[0]["variant"],
            "epoch": cand[0]["epoch"],
            "egm_variant": cand[0].get("egm_variant"),
            "egm_step": cand[0].get("egm_step"),
            "validation_decoder_mean_log_likelihood":
                cand[0]["validation_decoder_mean_log_likelihood"],
        }
        for setting, cand in payload["selected_top3_by_z_dim"].items()
    }
    return {
        "dataset": dataset,
        "arm": "decoderll",
        "criterion": CRITERION,
        "test_points_used_for_selection": 0,
        "secondary_root": str(secondary_root(dataset)),
        "selection_frozen": True,
        "selected": rows,
        "test_jobs": decoderll_task_count(dataset),
    }


LEGACY_SUFFIX = "-legacyq"


DECODERLL_SUFFIX = "-legacyq-decoderll"


REGISTRY: dict[str, dict[str, Any]] = {
    "Pendigits10": {"runner": "expansion", "run_name": "conditional-bgmbs-full-v1"},
}


LEGACYQ_DATASETS = tuple(REGISTRY)


def legacy_root(dataset: str, candidates: str = "primary") -> Path:
    suffix = LEGACY_SUFFIX if candidates == "primary" else DECODERLL_SUFFIX
    return HERE / "outputs" / dataset / (REGISTRY[dataset]["run_name"] + suffix)


def selection_source(dataset: str, candidates: str) -> Path:
    if candidates == "primary":
        return primary_root(dataset) / "frozen_iterative_top3.json"
    return secondary_root(dataset) / "frozen_iterative_top3.json"


def _legacy_runner(dataset: str, candidates: str = "primary"):
    install_legacy_proposal()
    if REGISTRY[dataset]["runner"] == "shared":
        inner = load_shared("legacyq_shared_runner", "protocol_conditional.py")
    else:
        inner = get_runner(dataset)
    inner.TOP_K = 1
    root = legacy_root(dataset, candidates)
    inner.output_root = lambda ds, _root=root: _root
    return inner


def legacy_stage(dataset: str, candidates: str = "primary") -> dict[str, Any]:
    primary = primary_root(dataset)
    root = legacy_root(dataset, candidates)
    root.mkdir(parents=True, exist_ok=True)

    egm_src = primary / "frozen_egm_selection.json"
    if egm_src.exists() and not (root / egm_src.name).exists():
        shutil.copyfile(egm_src, root / egm_src.name)

    frozen = root / "frozen_iterative_top3.json"
    if not frozen.exists():
        payload = json.loads(selection_source(dataset, candidates).read_text(encoding="utf-8"))
        key = "selected_top3_by_z_dim"
        payload[key] = {z: rows[:1] for z, rows in payload[key].items()}
        payload["truncated_to_rank1_for_legacy_q_rerun"] = True
        payload["proposal_convention"] = LEGACY_PROPOSAL_CONVENTION
        payload["selection_source"] = str(selection_source(dataset, candidates))
        payload["candidate_source"] = candidates
        frozen.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return json.loads(frozen.read_text(encoding="utf-8"))


def legacyq_task_count(dataset: str, candidates: str = "primary") -> int:
    runner = _legacy_runner(dataset, candidates)
    spec = runner.DATASET_SPECS[dataset]
    return len(spec["z_epsilon"]) * 1 * int(spec["test_shards"])


def legacyq_validate(dataset: str, candidates: str = "primary") -> dict[str, Any]:
    payload = legacy_stage(dataset, candidates)
    runner = _legacy_runner(dataset, candidates)
    spec = runner.DATASET_SPECS[dataset]
    return {
        "dataset": dataset,
        "arm": "legacyq",
        "proposal_convention": LEGACY_PROPOSAL_CONVENTION,
        "candidate_source": candidates,
        "legacy_root": str(legacy_root(dataset, candidates)),
        "selection_source": payload.get("selection_source"),
        "z_epsilon": [list(v) for v in spec["z_epsilon"]],
        "test_shards": int(spec["test_shards"]),
        "test_jobs": legacyq_task_count(dataset, candidates),
        "candidates": {z: c[0]["variant"] + "@" + str(c[0]["epoch"])
                       for z, c in payload["selected_top3_by_z_dim"].items()},
    }


def _primary_runner(dataset: str) -> ModuleType:
    dataset = str(dataset)
    if dataset == "BANK":
        inner = load_shared("bank_primary_runner", "protocol_unconditional.py")
        isolate_base(inner)
        inner.DATASET = "BANK"
        inner.X_DIM = 17
        inner.Z_DIMS = (8,)
        inner.DEFAULT_TEST_EPSILON = 0.0
        inner.PROPOSAL_MODES = ("legacy_product_q",)
        inner.OUTPUT_ROOT = primary_root(dataset)
        inner.base.load_split = core.load_split
        return inner
    if dataset == "ParkinsonsTelemonitoring":
        return runner
    return get_runner(dataset)


def _train_primary(dataset: str) -> ModuleType:
    inner = _primary_runner(dataset)
    if dataset in UNCONDITIONAL:
        for task in range(len(inner.egm_jobs())):
            inner.run_egm(task)
        inner.select_egm()
        for task in range(len(inner.iterative_jobs())):
            inner.run_iterative(task)
        return inner

    for task in range(len(inner.egm_jobs(dataset))):
        inner.run_egm(dataset, task)
    inner.select_egm(dataset)
    for task in range(len(inner.iterative_jobs(dataset))):
        inner.run_iterative(dataset, task)
    return inner


def reproduce(dataset: str) -> dict[str, Any]:
    dataset = "ParkinsonsTelemonitoring" if dataset.lower() == "parktele" else dataset
    if dataset not in (*UNCONDITIONAL, *SPECS):
        raise ValueError(f"Unknown dataset {dataset!r}")

    if dataset == "BANK":
        core.load_split(dataset)
    else:
        data.prepare(dataset)
    inner = _train_primary(dataset)

    if dataset in UNCONDITIONAL:
        score(dataset)
        select(dataset)
        for task in range(decoderll_task_count(dataset)):
            decoderll_test_shard(dataset, task)
        return decoderll_merge_test(dataset)

    if dataset == "Pendigits10":
        inner.select_iterative(dataset)
        score(dataset)
        select(dataset)
        legacy_stage(dataset, "decoderll")
        legacy = _legacy_runner(dataset, "decoderll")
        for task in range(legacyq_task_count(dataset, "decoderll")):
            legacy.run_test_shard(dataset, task)
        return legacy.merge_test(dataset)

    inner.select_iterative(dataset)
    for task in range(inner.test_task_count(dataset)):
        inner.run_test_shard(dataset, task)
    return inner.merge_test(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arms = parser.add_subparsers(dest="arm", required=True)

    stages = ("validate", "smoke", "egm", "select-egm", "iterative",
              "select-iterative", "test-shard", "merge-test", "merge-single-test")

    park = arms.add_parser("parktele", help="ParkinsonsTelemonitoring, unconditional")
    park.add_argument("command", choices=stages)
    park.add_argument("--task", type=int)

    cond = arms.add_parser("conditional", help="Pendigits10, EEGEye or Vehicle")
    cond.add_argument("command", choices=stages)
    cond.add_argument("--dataset", required=True, choices=sorted(SPECS))
    cond.add_argument("--task", type=int)

    first = arms.add_parser("decoderll", help="select the iterative checkpoint by validation decoder LL")
    first.add_argument("command", choices=("score", "select", "validate", "test-shard", "merge-test"))
    first.add_argument("--dataset", required=True, choices=DECODERLL_DATASETS)
    first.add_argument("--task", type=int)

    second = arms.add_parser("legacyq", help="re-score the test stage under the reported proposal convention")
    second.add_argument("command", choices=("validate", "test-shard", "merge-test"))
    second.add_argument("--dataset", required=True, choices=LEGACYQ_DATASETS)
    second.add_argument("--task", type=int)
    second.add_argument("--candidates", choices=("primary", "decoderll"), default="decoderll")

    full = arms.add_parser("reproduce", help="run the configured BayesNDE experiments")
    selection = full.add_mutually_exclusive_group()
    selection.add_argument("--dataset",
                      choices=("BANK", "ParkinsonsTelemonitoring", "parktele", *sorted(SPECS)))
    selection.add_argument("--all", action="store_true", help="run every configured UCI data set")
    full.add_argument("--config", type=Path, default=REPRODUCTION_CONFIG)

    args = parser.parse_args()
    if args.arm == "reproduce":
        if args.all or args.dataset is None:
            configured = yaml.safe_load(args.config.read_text(encoding="utf-8"))["uci"]["datasets"]
            result = {dataset: reproduce(dataset) for dataset in configured}
        else:
            result = reproduce(args.dataset)
        print(json.dumps(result, indent=2, default=str), flush=True)
        return
    if args.command == "test-shard" and args.task is None:
        parser.error("test-shard requires --task")

    if args.arm == "parktele":
        sys.argv = [sys.argv[0], args.command] + (["--task", str(args.task)] if args.task is not None else [])
        runner.main()
        return
    if args.arm == "conditional":
        sys.argv = ([sys.argv[0], args.command, "--dataset", args.dataset]
                    + (["--task", str(args.task)] if args.task is not None else []))
        get_runner(args.dataset).main()
        return

    if args.arm == "decoderll":
        if args.command == "score":
            payload = score(args.dataset)
            result = {k: v for k, v in payload.items() if k != "rows_by_setting"}
            result["settings"] = list(payload["rows_by_setting"])
        elif args.command == "select":
            payload = select(args.dataset)
            result = {k: v for k, v in payload.items() if k != "selected_top3_by_z_dim"}
            result["selected"] = {
                setting: {"variant": rows[0]["variant"], "epoch": rows[0]["epoch"],
                          "validation_decoder_mean_log_likelihood":
                              rows[0]["validation_decoder_mean_log_likelihood"]}
                for setting, rows in payload["selected_top3_by_z_dim"].items()
            }
        elif args.command == "validate":
            result = decoderll_validate(args.dataset)
        elif args.command == "test-shard":
            result = decoderll_test_shard(args.dataset, int(args.task))
        else:
            result = decoderll_merge_test(args.dataset)
    else:
        if args.command == "validate":
            result = legacyq_validate(args.dataset, args.candidates)
        else:
            legacy_stage(args.dataset, args.candidates)
            inner = _legacy_runner(args.dataset, args.candidates)
            result = (inner.run_test_shard(args.dataset, int(args.task))
                      if args.command == "test-shard" else inner.merge_test(args.dataset))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
