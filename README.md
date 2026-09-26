# BayesNDE

BayesNDE (Bayesian Neural Density Estimation) is a deep generative neural density
estimator. A Bayesian generative model with decoder $p_\theta(x \mid z)$ and prior
$p(z)$ is trained in two stages, an encoding-generative warm start (EGM) followed by
iterative updating, and the density of an observation is the normalizing constant

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz ,
$$

which is estimated with posterior sampling (HMC) and bridge sampling. This repository
provides the source code and the instructions to reproduce the BayesNDE results of the
paper on simulation data and real data.

## Table of Contents

- [Requirements](#requirements)
- [Install](#install)
- [Reproduction](#reproduction)
    - [Simulation Data](#simulation-data)
    - [Real Data](#real-data)
        - [UCI Datasets](#uci-datasets)
        - [Image Datasets](#image-datasets)
    - [Outlier Detection](#outlier-detection)
- [License](#license)

## Requirements

- Python 3.10
- TensorFlow 2.15.1
- TensorFlow Probability 0.23.0

The complete list of pinned versions is in `requirements.txt`. A GPU is recommended for
training.

## Install

```shell
git clone https://github.com/liuq-lab/BayesNDE.git
cd BayesNDE
pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The code is organised as follows.

```text
bayesgm/        generative model and neural networks
bayesnde/       training stages, generation diagnostics, bridge-sampling estimators
configs/        shared model, estimator and experiment-set configurations
experiments/    one driver per experiment (simulation, uci, application_mnist, application_odds)
```

## Reproduction

This section explains how to reproduce the BayesNDE results in the paper. All commands
are run from the repository root. Each `reproduce` command downloads or simulates its
data, trains the model, selects the checkpoints on validation data only, and evaluates
the test data once the selection is frozen. Seeds, data splits and estimator settings
are fixed in the drivers and in `configs/`; the experiment sets (dimensions, data sets,
seeds) are listed in `configs/reproduction.yaml`. Data, checkpoints and results are
written below `outputs/` (UCI: `experiments/uci/outputs/`).

### Simulation Data

#### 1. Density visualization

Independent Gaussian mixture. The estimated and true densities are evaluated on a
50 x 50 grid.

```shell
python -m experiments.simulation.indep_gmm visualization reproduce
```

The density figures and `metrics.json` are written to
`outputs/simulation/independent_gmm_visualization/test/result/`.

Involute. The iterative-updating learning rate and epoch are selected jointly by the
generation rank on validation data.

```shell
python -m experiments.simulation.involute --stage reproduce
```

The selection and the grid evaluation are written to `outputs/simulation/involute/`.

#### 2. Independent Gaussian mixture dimension scaling

Every dimension is scored by the Spearman correlation between the estimated and the true
density on 2,000 held-out observations.

For low dimension (p ≤ 10):

```shell
python -m experiments.simulation.indep_gmm lowdim reproduce
```

For high dimension (p > 10):

```shell
python -m experiments.simulation.indep_gmm highdim reproduce
```

Results are written below `outputs/simulation/independent_gmm_dimension_scaling/` and
`outputs/simulation/independent_gmm_highdim/`, one `dim_XXX/test/result/metrics.json`
per dimension. A single dimension can be run with `--dim`.

### Real Data

#### UCI Datasets

We evaluate the average test log-likelihood on BANK, ParkinsonsTelemonitoring (an
unconditional density $p(x)$) and on Pendigits10, Vehicle and EEGEye (a class-conditional
density $p(x \mid y)$). The data are downloaded from their public sources and checksum
verified.

```shell
python -m experiments.uci.run reproduce --all
```

A single data set can be run with `--dataset`, e.g. `--dataset Vehicle`. The per-point
test log-likelihoods (`full_test_log_likelihood.npz`) and their mean (`metrics.json`) are
written below `experiments/uci/outputs/<dataset>/`; see
[`experiments/uci/README.md`](experiments/uci/README.md) for the data definitions.

#### Image Datasets

MNIST is modelled with ten class-conditional densities $p(x \mid y)$, and a test image is
classified as $\arg\max_y \log p(x \mid y)$ under a uniform class prior.

```shell
python -m experiments.application_mnist.run_bgm reproduce
```

The test accuracy is written to `outputs/mnist_v2/bernoulli_logit/seed_0/metrics.json`.
Conditionally generated images can be drawn with
`python -m experiments.application_mnist.visualize_bgm`.

### Outlier Detection

Shuttle from the ODDS library. BayesNDE fits the density of the training data, ranks the
test observations by their estimated log-density and reports precision at k, where k is
the number of anomalies in the test split.

```shell
python -m experiments.application_odds.run_shuttle
```

The metrics of each run and their summary (`summary.json`) are written below
`outputs/shuttle/`.

## License

This project is licensed under the MIT License - see [`LICENSE`](LICENSE) for details.
