# Chapter 14 — Deployment & Quantization

Getting a trained model onto a constrained device. A microcontroller has no
filesystem and little memory, so this chapter covers the tricks that make
deployment fit: baking weights into the binary as bytes, keeping inference code
generic over the backend, shrinking weights with int8 quantization, and counting
the footprint before you ship.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `bytes1` | weights in RAM — record to `Vec<u8>`, load back with `load_record` | `ch14` · e01 weights baked into the binary |
| `generic1` | inference generic over the backend `B` | `ch14` · e02 inference generic over the backend |
| `quant1` | int8 quantization by hand — `round(w/scale)` | `ch14` · e03 int8 quantisation by hand |
| `count1` | model footprint — params × bytes-per-value | `ch14` · e04 counting parameters and bytes |

Run one with `cargo run --example bytes1`, check it with `cargo test --example bytes1`.
Three are logic errors caught by a test (an unapplied record, a truncating cast
instead of a round, an f16 size computed with the f32 factor); `generic1` is a
compile error — the call site asks for a backend the function isn't generic
over, so you make `run_inference` generic over `B`.

> Note: chapter 14 is on the learning-burn upstream `main`. Values here were
> confirmed by running its examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 14.
