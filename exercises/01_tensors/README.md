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
| `tensors1` | rank is part of the type | `ch1_01_rank_vs_shape.rs` |
| `tensors2` | creation; nesting sets the shape | `ch1_02_basic_creation.rs` |
| `tensors3` | integer tensors are a different kind | `ch1_03_int_tensors.rs` |
| `tensors4` | filled constructors (`full`) | `ch1_04_filled_structured.rs` |
| `tensors5` | random tensors & distributions | `ch1_05_random_tensors.rs` |
| `tensors6` | building a tensor from a struct | `ch1_06_from_custom_type.rs` |
| `tensors7` | the data bridge; `into_data` consumes | `ch1_07_tensor_data_bridge.rs` |
| `tensors8` | ownership: clone when you reuse | `ch1_08_ownership_cloning.rs` |
| `tensors9` | float closeness vs exact equality | `ch1_09_display_and_closeness.rs` |

Run one with `cargo run --example tensors2`, check it with
`cargo test --example tensors2`. Five of these are compile errors, four are
logic errors caught by a test — the mix Rustlings uses too.

Read alongside *Learning Burn*, chapter 1.
