# Chapter 3 — Matmul & the Shape Rule

Matrix multiplication is where shapes stop being a formality. The rule is
`[m, k] @ [k, n] -> [m, n]`: the inner dimensions must match, and the result
takes the outer ones. This is a different operation from the element-wise `*`
of chapter 2, and Burn keeps them clearly separate — `matmul` for the matrix
product, `*` for element-wise.

Each exercise maps to a runnable example in the *Learning Burn* book's
`examples/chapter03/`:

| Exercise | Concept | Book example |
|---|---|---|
| `matmul1` | matrix product vs element-wise `*` | `e01_matmul` |
| `matmul2` | matrix × vector with `linalg::matvec` | `e02_transpose_matvec_outer` |
| `matmul3` | batched matmul and its shape rule | `e03_structured_and_batched` |

Run one with `cargo run --example matmul1`, check it with
`cargo test --example matmul1`. One is a compile error (`matmul2`), the other
two are logic errors caught by a test.

Read alongside *Learning Burn*, chapter 3.
