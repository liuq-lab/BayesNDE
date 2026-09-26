# Configuration files

- `model.yaml`: shared neural-network and training defaults.
- `bridge.yaml`: shared bridge-sampling and posterior-sampling defaults.
- `reproduction.yaml`: experiment groups, data sets, dimensions, and random seeds used by the public reproduction commands.

Experiment drivers materialize a complete resolved configuration in their output
directory. Those generated files are the authoritative record for an individual run.
