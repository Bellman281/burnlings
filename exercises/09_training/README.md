# Chapter 9 — Training

Where it all comes together: a real training loop that fits a model with an
optimizer. Burn's loop has **four beats** — forward, loss, backward, step — and
its optimizer API is **functional**: `optim.step(...)` consumes the model and
returns a new, updated one, so you reassign it (`model = optim.step(...)`).
Swapping optimizers (SGD, momentum, Adam, AdamW) changes only one config line;
the loop stays identical.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `sgd1` | the four-beat SGD loop; `step` returns the new model | `ch9` · a real training loop |
| `opt1` | configuring an optimizer (SGD + momentum) | `ch9` · choosing an optimizer |

Run one with `cargo run --example sgd1`, check it with `cargo test --example sgd1`.
Both are compile errors this time — `sgd1` forgets to reassign the model after
`step` (so it's moved away and never learns), and `opt1` forgets to wrap the
momentum config in `Some`.

> Note: chapter 9 is on the learning-burn upstream `main`. Verified on Burn 0.21
> — the trained model predicts `f(5) ≈ 11`.

Read alongside *Learning Burn*, chapter 9.
