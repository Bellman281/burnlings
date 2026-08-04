// attn1.rs — Burnlings · Chapter 12: Attention & Transformers
//
// The one equation behind every transformer:
//
//      Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V
//
// The `1 / sqrt(d_k)` scale keeps the dot products from growing with the feature
// dimension and shoving softmax into its flat, tiny-gradient tails. Drop the
// scale and the weights — and the output — come out wrong.
//
// TODO: Scale the scores by 1/sqrt(d_k) before the softmax
//       (`.mul_scalar(1.0 / d_k.sqrt())`).
//       (Compare book example chapter12/e01_scaled_dot_product_attention.)
//
// I AM NOT DONE

use burn::backend::Flex;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;

type Backend = Flex;

fn attention() -> Vec<f32> {
    let device = Default::default();
    // seq_len = 2, d_k = 2. Q and K are the identity so the scores are easy to read.
    let q = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let k = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let v = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let d_k = q.dims()[1] as f64;

    // ⬇️ unscaled: the 1/sqrt(d_k) factor is missing, so the softmax is too sharp
    let scores = q.matmul(k.transpose());
    let _ = d_k;

    let weights = softmax(scores, 1);
    let out = weights.matmul(v);
    out.into_data().to_vec().unwrap()
}

fn main() {
    println!("attention output = {:?}", attention());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaled_dot_product() {
        let out = attention();
        let expect = [1.6604769, 2.660477, 2.339523, 3.339523];
        for (o, e) in out.iter().zip(expect.iter()) {
            assert!((o - e).abs() < 1e-3, "got {o}, want {e}");
        }
    }
}
