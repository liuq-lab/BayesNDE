# Shuttle anomaly detection

BayesNDE fits the normal-data density and ranks low-density test observations as
anomalies. The reported endpoint is precision@k, where k is the number of
anomalies in the test split. Labels do not enter training or checkpoint
selection.

## Reproduction

Run from the repository root:

```bash
python -m experiments.application_odds.run_shuttle
```

The downloaded archive and Shuttle file are checksum-verified. The fixed
preprocessing, split sizes, and hashes are recorded in `config.json`; experiment
seeds are defined in `configs/reproduction.yaml`. Outputs are written below
`outputs/shuttle/`.
