# Chapter 1 — Tensors

The tensor is Burn's core data structure. The first thing to internalise, and
the thing that makes Burn different from NumPy or PyTorch, is that a tensor's
**rank** — its number of dimensions — is part of its **type**.

`Tensor<B, 1>` (a vector) and `Tensor<B, 2>` (a matrix) are *different types*.
The compiler will not let you accidentally mix them. A shape bug that would blow
up at runtime in Python becomes a compile error here.

- `tensors1` — rank is part of the type. Corresponds to the book's
  `ch1_01_rank_vs_shape.rs`.

Read alongside *Learning Burn*, chapter 1.
