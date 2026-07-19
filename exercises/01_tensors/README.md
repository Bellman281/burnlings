# Chapter 1 — Tensors

The tensor is Burn's core data structure. The first thing to internalise — and
the thing that makes Burn different from NumPy or PyTorch — is that a tensor's
**rank** (its number of dimensions) is part of its **type**. `Tensor<B, 1>` (a
vector) and `Tensor<B, 2>` (a matrix) are *different types*; the compiler won't
let you accidentally mix them. Shape bugs that would blow up at runtime in
Python become compile errors here.

Each exercise below maps to a runnable example in the *Learning Burn* book's
`book-tests/examples/`:

| Exercise | Concept | Book example |
|---|---|---|
| `tensors1` | rank is part of the type | `e01_rank_vs_shape` |
| `tensors2` | creation; nesting sets the shape | `e02_basic_creation` |
| `tensors3` | integer tensors are a different kind | `e03_integer_tensors` |
| `tensors4` | filled constructors (`full`) | `e04_tensors_filled_with_known_values` |
| `tensors5` | random tensors & distributions | `e05_random_tensors_from_two_distributions` |
| `tensors6` | building a tensor from a struct | `e06_a_tensor_from_your_own_struct` |
| `tensors7` | the data bridge; `into_data` consumes | `e07_getting_your_numbers_back_out` |
| `tensors8` | ownership: clone when you reuse | `e08_min_max_normalisation` |
| `tensors9` | float closeness vs exact equality | `e09_printing_and_comparing_tensors_safely` |

Run one with `cargo run --example tensors2`, check it with
`cargo test --example tensors2`. Five of these are compile errors, four are
logic errors caught by a test — the mix Rustlings uses too.

Read alongside *Learning Burn*, chapter 1.
