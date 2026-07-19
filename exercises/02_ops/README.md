# Chapter 2 — Tensor Ops

Once you can build tensors (chapter 1), you compute with them. This chapter is
the everyday vocabulary: element-wise arithmetic, broadcasting, reshaping and
slicing, reductions, feature standardisation, and boolean masking. The recurring
theme is that Burn makes you say out loud what NumPy/PyTorch do implicitly —
most visibly with broadcasting, where you must `unsqueeze()` a vector before it
lines up against a matrix.

Each exercise maps to a runnable example in the *Learning Burn* book's
`examples/chapter02/`:

| Exercise | Concept | Book example |
|---|---|---|
| `ops1` | element-wise arithmetic (`*` is not matmul) | `e01_element_wise_arithmetic` |
| `ops2` | broadcasting via `unsqueeze()` | `e02_broadcasting_with_unsqueeze` |
| `ops3` | `reshape` + `slice` (half-open ranges) | `e03_reshape_flatten_slice` |
| `ops4` | dimension reductions keep the axis | `e04_reductions_sum_mean_max` |
| `ops5` | z-score standardise (std = √var) | `e05_standardising_features` |
| `ops6` | boolean masks + `mask_where` | `e06_maths_clamp_mask` |

Run one with `cargo run --example ops1`, check it with
`cargo test --example ops1`. One is a compile error (`ops2`), the other five are
logic errors caught by a test.

Read alongside *Learning Burn*, chapter 2.
