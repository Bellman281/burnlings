# Chapter 11 — A Network From Scratch

No `nn` layers, no `Optimizer` — just tensors, matmuls, and the four beats of
training written out by hand. This chapter rebuilds everything the earlier
modules gave you for free: a layer is an `x @ W` matmul, a hand-derived gradient
must match autodiff to the last digit, and the SGD update is a literal
`param <- param - lr * grad`. It's the "aha" chapter where the framework stops
being magic.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `net1` | a layer is a matmul — two layers, `out = relu(x@W1)@W2` | `ch11` · e01 a layer is a matmul |
| `net2` | gradients by hand vs autodiff — `dW1 = xᵀ @ dh_pre` | `ch11` · e02 gradients by hand and by autodiff |
| `net3` | training from scratch — manual SGD via `from_inner`/`inner` | `ch11` · e03 training from scratch |

Run one with `cargo run --example net1`, check it with `cargo test --example net1`.
All three are logic errors caught by a test: `net1` returns the hidden layer
instead of applying the output layer, `net2` forgets to transpose `x` in the
weight gradient (so hand-backprop stops matching autodiff), and `net3` *adds*
the gradient (ascent) where descent must *subtract* it.

> Note: chapter 11 is on the learning-burn upstream `main`. The seeded runs are
> deterministic (`Backend::seed`); values here were confirmed by running the
> book's examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 11.
