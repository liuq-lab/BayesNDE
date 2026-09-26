#!/usr/bin/env python3
"""Summarize the three reported Shuttle runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "shuttle"
CONFIG = REPO / "configs" / "reproduction.yaml"


def main(config_path: Path = CONFIG) -> None:
    seeds = tuple(int(seed) for seed in yaml.safe_load(
        config_path.read_text(encoding="utf-8"))["shuttle"]["seeds"])
    rows = []
    for seed in seeds:
        path = ROOT / f"seed_{seed}" / "arms" / "egm_genrank_rm" / "metrics_decoderll.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            "precision_at_k": float(payload["precision_at_k"]),
            "metrics_path": str(path.resolve()),
        })
    values = np.asarray([row["precision_at_k"] for row in rows], dtype=np.float64)
    summary = {
        "dataset": "Shuttle",
        "method": "BayesNDE",
        "seeds": list(seeds),
        "rows": rows,
        "mean_precision_at_k": float(np.mean(values)),
        "population_standard_deviation": float(np.std(values, ddof=0)),
    }
    output = ROOT / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
