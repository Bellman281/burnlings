# Burnlings 🔥🦀

Small exercises to get you used to reading and writing deep-learning code in
Rust with the [Burn](https://github.com/tracel-ai/burn) framework — in the
spirit of [Rustlings](https://github.com/rust-lang/rustlings).

Each exercise is a single file with a compiler or logic error. Your job is to
fix it. The exercises track the chapters of the *Learning Burn* book, so you can
read a chapter and then drill the idea until it compiles.

## How it works

Every exercise starts with an `// I AM NOT DONE` comment and a `// TODO`
describing what to fix. Fix the code, remove the `I AM NOT DONE` line, and move
on. Metadata (order, hints, whether an exercise is checked by a test) lives in
[`info.toml`](info.toml), the same format Rustlings uses.

```
burnlings/
├── Cargo.toml            # burn 0.21.0, ndarray + autodiff backend
├── info.toml             # exercise list + hints (Rustlings format)
├── exercises/
│   └── 01_tensors/
│       └── tensors1.rs   # <- the exercise you solve
└── solutions/
    └── 01_tensors/
        └── tensors1.rs   # <- reference solution (peek only if stuck)
```

## Try the sample exercise

```bash
cargo run  --example tensors1     # runs it (fails until you fix the rank)
cargo test --example tensors1     # checks it with the built-in test
```

`tensors1` is broken on purpose — it won't compile until you fix it. That's the
point. If you get stuck, read the `hint` for `tensors1` in `info.toml`, or open
`solutions/01_tensors/tensors1.rs`.

## Relationship to Learning Burn

Exercises mirror the runnable examples in the
[Learning Burn](https://github.com/jhosein58/learning-burn) book's
`book-tests/examples/`. Chapter 1 (`tensors1` here) corresponds to
`ch1_01_rank_vs_shape.rs`: rank is part of a Burn tensor's type.

## Credit

Burn is built by the [Tracel AI team](https://github.com/tracel-ai/burn).
This repo is exercise code against Burn's public API, patterned after Rustlings.
