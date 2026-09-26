# UCI density estimation

This directory reproduces the five BayesNDE UCI results reported in the paper.
The experiment driver handles data preparation, training, validation-only model
selection, and final test evaluation.

## Data sets

| Data set | Dimension | Target |
| :-- | --: | :-- |
| BANK | 17 | $p(x)$ |
| ParkinsonsTelemonitoring | 14 | $p(x)$ |
| Pendigits10 | 16 | $p(x \mid y)$ |
| Vehicle | 18 | $p(x \mid y)$ |
| EEGEye | 14 | $p(x \mid y)$ |

Downloaded files, processed splits, and fingerprints are cached locally. The
data loader verifies the source files and uses the fixed preprocessing and
splits encoded in `data.py`.

## Reproduction

Run from the repository root:

```bash
python -m experiments.uci.run reproduce --all
```

The EGM warm start is selected with held-out generation diagnostics. The
iterative checkpoint is selected by validation decoder log likelihood. The test
split is scored only after both selections are frozen.

Final arrays are named `full_test_log_likelihood.npz`; the associated JSON
files record the mean, selected checkpoints, data fingerprint, and estimator
configuration. All artifacts are created below `experiments/uci/outputs/`.
