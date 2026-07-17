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
│       ├── tensors1.rs … tensors9.rs   # <- the exercises you solve
│       └── README.md
└── solutions/
    └── 01_tensors/
        └── tensors1.rs … tensors9.rs   # <- reference (peek only if stuck)
```

## Chapter 1 — tensors (9 exercises)

```bash
cargo run  --example tensors1     # runs it (fails until you fix it)
cargo test --example tensors1     # checks it with the built-in test
```

Every exercise is broken on purpose — it won't pass until you fix it. Five are
compile errors, four are logic errors caught by a test. Work through
`tensors1` → `tensors9`; they follow the *Learning Burn* book's chapter 1, one
concept each (rank, creation, int tensors, filled constructors, random,
building from a struct, the data bridge, ownership/cloning, float closeness).
See `exercises/01_tensors/README.md` for the exercise-to-book map.

Stuck? Read the `hint` for that exercise in `info.toml`, or open the matching
file under `solutions/01_tensors/`.

## Chapter 2 — tensor ops (6 exercises)

```bash
cargo run  --example ops1
cargo test --example ops1
```

Computing with tensors: element-wise arithmetic, broadcasting (`unsqueeze`),
reshaping and slicing, reductions, feature standardisation, and boolean masking.
Work through `ops1` → `ops6`; they follow the *Learning Burn* book's chapter 2.
One is a compile error, five are logic errors caught by a test. See
`exercises/02_ops/README.md` for the exercise-to-book map.

## Relationship to Learning Burn

Exercises mirror the runnable examples in the
[Learning Burn](https://github.com/jhosein58/learning-burn) book's
`book-tests/examples/`. Chapter 1 (`tensors1` here) corresponds to
`ch1_01_rank_vs_shape.rs`: rank is part of a Burn tensor's type.

## Credit

Burn is built by the [Tracel AI team](https://github.com/tracel-ai/burn).
This repo is exercise code against Burn's public API, patterned after Rustlings.
