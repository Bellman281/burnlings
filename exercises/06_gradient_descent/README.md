# Chapter 6 — Gradient Descent

Learning by descent — fitting a model by repeatedly nudging the weights *against*
the loss gradient until they settle. First the long way (derive the mean-squared-
error gradient by hand and subtract it), then the short way (let the autodiff of
chapter 5 compute the very same gradient for you).

Each exercise maps to a runnable example in the *Learning Burn* book's
`book-tests/examples/`:

| Exercise | Concept | Book example |
|---|---|---|
| `gd1` | manual gradient descent — fit `y = 2x + 1` → `w = [1, 2]` | `ch6_01_gradient_descent` |
| `gd2` | MSE gradient via autodiff — loss 41, grad `[-12, -35]` | `ch6_02_autodiff_gradient` |

Run one with `cargo run --example gd1`, check it with `cargo test --example gd1`.
Both are logic errors caught by a test.

> Note: chapter 6 lives on the learning-burn `examples/chapter-6` branch (not yet
> merged to that repo's `main`). Values here were confirmed by running the book's
> examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 6.
