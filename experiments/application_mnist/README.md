# MNIST conditional-density classification

BayesNDE trains ten class-conditional image densities and predicts
`argmax_y log p(x | y)` under a uniform class prior. Test labels are used only
to compute the final accuracy.

The observation model is Bernoulli with a logistic-normal moment approximation:

```text
z ~ N(0, I_10)
logits, variance = G(z, y)
x | z, y ~ Bernoulli(sigmoid(logits / sqrt(1 + pi * variance / 8)))
```

The same closed-form likelihood is used during iterative training, posterior
sampling, and bridge evaluation.

## Reproduction

Run from the repository root:

```bash
python -m experiments.application_mnist.run_bgm reproduce
```

The fixed split contains 55,000 training, 5,000 validation, and 10,000 test
images. Checkpoint selection uses only the validation set. Outputs are written
below `outputs/mnist_v2/bernoulli_logit/`.

Optional conditional-generation figures can be produced with:

```bash
python -m experiments.application_mnist.visualize_bgm
```
