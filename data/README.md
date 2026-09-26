# Data

This directory is intentionally distributed without datasets. Local datasets,
raw archives, processed splits, and generated caches are excluded from version
control for data-privacy reasons.

Experiment scripts may download or create their required files here. In
particular, the UCI expansion workflow uses `data/uci_expansion_v1/`; keep that
directory local and do not force-add it to Git.

See the README for the relevant experiment under `experiments/` for data
preparation instructions.
