# Chapter 7 — Activations

The non-linearities that let a network learn more than a straight line. This
chapter covers the three families you meet everywhere: the ReLU family (cheap,
the default for hidden layers), the saturating pair (sigmoid and tanh, which
squash into bounded ranges), and softmax (which turns logits into a probability
distribution). Burn keeps these in `burn::tensor::activation`.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `act1` | ReLU zeroes every negative (vs leaky ReLU) | `ch7` · the ReLU family |
| `act2` | sigmoid → 0.5 at 0 (vs tanh → 0) | `ch7` · sigmoid & tanh |
| `act3` | softmax over the class axis (dim 1, not dim 0) | `ch7` · softmax |

Run one with `cargo run --example act1`, check it with `cargo test --example act1`.
All three are logic errors caught by a test — `act3` is the classic "softmaxed
the wrong axis" bug that compiles cleanly but is wrong.

> Note: chapter 7 is on the learning-burn upstream `main` (per-example layout);
> values here were confirmed by running its examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 7.
