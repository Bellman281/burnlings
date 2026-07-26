# Chapter 13 — Saving & Reloading

Training and inference are usually separate programs: you train once, write the
weights to disk, and a lightweight process reloads them to serve predictions.
This chapter covers that round trip and the choice of *recorder* — the object
that decides how weights are serialized, and at what numeric precision.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `save1` | train → save → reload → predict (`save_file` / `load_file`) | `ch13` · e01 train, save, reload, predict |
| `prec1` | recorder precision — full (f32) is exact, compact (f16) rounds | `ch13` · e02 same weights, two precisions |

Run one with `cargo run --example save1`, check it with `cargo test --example save1`.
Both are logic errors caught by a test: `save1` predicts from a fresh model
because the saved weights were never loaded back in, and `prec1` uses the
compact (f16) recorder where an exact round trip needs full precision.

> Note: chapter 13 is on the learning-burn upstream `main`. Values here were
> confirmed by running its examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 13.
