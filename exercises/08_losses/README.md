# Chapter 8 — Losses

The objective functions a model minimises. This chapter covers six, grouped by
task: **MSE** and **Huber** for regression, **cross-entropy** and **binary
cross-entropy** for classification, **KL divergence** for matching distributions,
and **cosine embedding** for direction. Burn's losses live in `burn::nn::loss`
and mostly follow a config/instance pattern:
`SomeLossConfig::new()…init().forward(…)`.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Loss | Book example |
|---|---|---|
| `mse1` | mean squared error (mean vs sum reduction) | `e01_mean_squared_error` |
| `ce1` | cross-entropy — raw logits, no double-softmax | `e02_cross_entropy` |
| `huber1` | Huber — robust regression (delta) | `e03_huber_loss` |
| `bce1` | binary cross-entropy — `with_logits` | `e04_binary_cross_entropy` |
| `kldiv1` | KL divergence — log-probs vs probs | `e05_kl_divergence` |
| `cosine1` | cosine embedding — `+1/-1` targets | `e06_cosine_embedding` |

Run one with `cargo run --example mse1`, check it with `cargo test --example mse1`.
All six are logic errors caught by a test — several are the "compiles fine but
silently wrong" traps the book warns about: pre-softmaxing before cross-entropy,
forgetting `with_logits`, or using `softmax` where KL divergence needs
`log_softmax`.

> Note: chapter 8 is on the learning-burn upstream `main`. Values here were
> confirmed by running its examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 8.
