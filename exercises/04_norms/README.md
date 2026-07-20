# Chapter 4 — Norms & Gram

Length and inner-product structure. The **L2 norm** measures a vector's length;
**normalising** divides by it to land on the unit circle; and the **Gram matrix**
`X^T X` collects the pairwise inner products of a design matrix's columns — the
quantity at the heart of least-squares.

Each exercise maps to a runnable example in the *Learning Burn* book's
`book-tests/examples/`:

| Exercise | Concept | Book example |
|---|---|---|
| `norms1` | L2 (Euclidean) norm | `ch4_01_vector_norm` |
| `norms2` | normalise to a unit vector | `ch4_01_vector_norm` |
| `gram1` | Gram matrix `X^T X` | `ch4_02_gram_matrix` |

Run one with `cargo run --example norms1`, check it with
`cargo test --example norms1`. All three are logic errors caught by a test.

> Note: the book's `ch4_01` also demonstrates `linalg::det`, which in Burn 0.21
> requires a batched (rank-3) input — it's left out here until that's resolved
> upstream (`fix/ch4-det-requires-batch-rank`).

Read alongside *Learning Burn*, chapter 4.
