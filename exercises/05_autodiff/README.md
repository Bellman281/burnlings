# Chapter 5 — Autodiff

Gradients — the reason a deep-learning framework exists. In Burn, autodiff is a
backend *decorator*: `Autodiff<NdArray>` wraps a backend so it records a
computation graph. You opt an input in with `.require_grad()`, call
`.backward()` on the output, and read each gradient back out of the container it
returns. For inference you peel the wrapper off with `.inner()` and drop to the
plain base backend — Burn has no `no_grad` block.

Each exercise maps to a runnable example in the *Learning Burn* book's
`book-tests/examples/`:

| Exercise | Concept | Book example |
|---|---|---|
| `grad1` | `require_grad` → `backward` → `grad` (df/dx of sum(x²) = 2x) | `e01_backward_your_first_gradient` |
| `grad2` | one container, look up each gradient (dL/da = b, dL/db = a) | `e02_the_gradient_container` |
| `grad3` | `inner()` drops autodiff for inference | `e03_inner_inference_without_tracking` |

Run one with `cargo run --example grad1`, check it with
`cargo test --example grad1`. `grad1` and `grad2` are logic errors caught by a
test; `grad3` is a compile error.

Read alongside *Learning Burn*, chapter 5.
