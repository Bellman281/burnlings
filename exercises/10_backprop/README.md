# Chapter 10 — Backprop

Backpropagation is just the chain rule run backwards through the graph. This
chapter takes autodiff apart and puts it back together: first checking a
gradient by hand against `.backward()`, then watching a gradient flow *through*
a hidden layer to the first weight matrix, and finally training a two-layer MLP
to solve XOR — the classic problem a single linear layer can never crack.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `bp1` | the chain rule, checked by hand — `loss = (a*b)²` → `da=36, db=24` | `ch10` · e01 the chain rule |
| `bp2` | backprop through a hidden layer — the first weight needs `.require_grad()` | `ch10` · e02 backprop through a hidden layer |
| `xor1` | learning XOR — a nonlinearity between two Linear layers | `ch10` · e03 learning XOR |

Run one with `cargo run --example bp1`, check it with `cargo test --example bp1`.
`bp1` and `xor1` are logic errors caught by a test; `bp2` panics at runtime
because a gradient never reaches the un-`require_grad`'d weight. `xor1` is the
big one — without the `tanh` between the layers, two Linear maps collapse into
one and the loss stalls near 0.25; add it and the net drives the loss to ~0.

> Note: chapter 10 is on the learning-burn upstream `main`. The seeded XOR run
> is deterministic (`Backend::seed`); values here were confirmed by running the
> book's examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 10.
