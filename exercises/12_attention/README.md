# Chapter 12 — Attention & Transformers

The architecture behind modern language models, built up from one equation. This
chapter assembles a transformer encoder block piece by piece: the scaled
dot-product attention at its core, the causal mask that makes it autoregressive,
the multi-head split that lets it attend several ways at once, and finally the
full block with residuals and LayerNorm. Everything is written against Burn's
plain tensor ops so nothing is a black box.

Each exercise maps to a runnable example in the *Learning Burn* book:

| Exercise | Concept | Book example |
|---|---|---|
| `attn1` | scaled dot-product attention — `softmax(QKᵀ/√d_k)V` | `ch12` · e01 scaled dot-product attention |
| `attn2` | causal masking — hide the future with a `-1e9` additive mask | `ch12` · e02 causal masking |
| `attn3` | multi-head attention — softmax over the keys axis | `ch12` · e03 multi-head attention |
| `attn4` | a transformer block — residual + LayerNorm ×2 | `ch12` · e04 a transformer block |

Run one with `cargo run --example attn1`, check it with `cargo test --example attn1`.
All four are logic errors caught by a test — the kind that compile fine but are
quietly wrong: an unscaled score, a flipped mask sign, softmax on the wrong axis,
and a missing final LayerNorm. `attn3` and `attn4` seed the backend so their
tests are deterministic.

> Note: chapter 12 is on the learning-burn upstream `main`. Values here were
> confirmed by running its examples on Burn 0.21.

Read alongside *Learning Burn*, chapter 12.
