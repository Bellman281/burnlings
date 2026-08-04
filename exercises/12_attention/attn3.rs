// attn3.rs — Burnlings · Chapter 12: Attention & Transformers
//
// Multi-head attention runs several scaled-dot-product attentions in parallel
// over head-sized slices of the features, then concatenates them. The scores
// tensor has shape [heads, seq, seq]; the softmax must normalise over the KEYS,
// which is the LAST dim (dim 2), so every query's attention weights sum to 1.
//
// Softmax the wrong axis and the rows no longer sum to 1 — a silent bug that
// still runs and still has the right shape.
//
// TODO: Softmax over the keys axis — the last dim, `2`, not `1`.
//       (Compare book example chapter12/e03_multi_head_attention.)
//
// I AM NOT DONE

use burn::backend::Flex;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend as _;
use burn::tensor::Tensor;

type Backend = Flex;

// Returns (weights dims, output dims, per-query weight sums over the key axis).
fn multi_head() -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let device = Default::default();
    Backend::seed(&device, 7);

    let seq = 3;
    let d_model = 4;
    let n_heads = 2;
    let d_k = d_model / n_heads; // 2

    let x = Tensor::<Backend, 2>::from_floats(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
        &device,
    );

    let wq: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wk: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wv: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wo: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);

    // split into heads: [seq, d_model] -> [seq, heads, d_k] -> [heads, seq, d_k]
    let q = wq.forward(x.clone()).reshape([seq, n_heads, d_k]).swap_dims(0, 1);
    let k = wk.forward(x.clone()).reshape([seq, n_heads, d_k]).swap_dims(0, 1);
    let v = wv.forward(x.clone()).reshape([seq, n_heads, d_k]).swap_dims(0, 1);

    let scale = 1.0 / (d_k as f64).sqrt();
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale); // [heads, seq, seq]

    // ⬇️ wrong axis: attention must normalise over the keys = the LAST dim (2)
    let weights = softmax(scores, 1);

    let context = weights.clone().matmul(v); // [heads, seq, d_k]
    let concat = context.swap_dims(0, 1).reshape([seq, d_model]);
    let out = wo.forward(concat);

    let sums = weights.clone().sum_dim(2).into_data().to_vec().unwrap();
    (weights.dims().to_vec(), out.dims().to_vec(), sums)
}

fn main() {
    let (wdims, odims, sums) = multi_head();
    println!("weights dims = {wdims:?}, out dims = {odims:?}");
    println!("weight row sums (should all be 1) = {sums:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn softmax_over_the_keys() {
        let (wdims, odims, sums) = multi_head();
        assert_eq!(wdims, vec![2, 3, 3], "weights should be [heads, seq, seq]");
        assert_eq!(odims, vec![3, 4], "a block preserves the input shape");
        for s in &sums {
            assert!((s - 1.0).abs() < 1e-3, "weight row sum {s} != 1 (wrong softmax axis?)");
        }
    }
}
