// attn3.rs — Burnlings · Chapter 12: Attention & Transformers (SOLUTION)
//
// Multi-head attention: softmax normalises over the keys (the last dim, 2).

use burn::backend::Flex;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend as _;
use burn::tensor::Tensor;

type Backend = Flex;

fn multi_head() -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let device = Default::default();
    Backend::seed(&device, 7);

    let seq = 3;
    let d_model = 4;
    let n_heads = 2;
    let d_k = d_model / n_heads;

    let x = Tensor::<Backend, 2>::from_floats(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
        &device,
    );

    let wq: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wk: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wv: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wo: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);

    let q = wq.forward(x.clone()).reshape([seq, n_heads, d_k]).swap_dims(0, 1);
    let k = wk.forward(x.clone()).reshape([seq, n_heads, d_k]).swap_dims(0, 1);
    let v = wv.forward(x.clone()).reshape([seq, n_heads, d_k]).swap_dims(0, 1);

    let scale = 1.0 / (d_k as f64).sqrt();
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);

    let weights = softmax(scores, 2);

    let context = weights.clone().matmul(v);
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
        assert_eq!(wdims, vec![2, 3, 3]);
        assert_eq!(odims, vec![3, 4]);
        for s in &sums {
            assert!((s - 1.0).abs() < 1e-3, "weight row sum {s} != 1");
        }
    }
}
