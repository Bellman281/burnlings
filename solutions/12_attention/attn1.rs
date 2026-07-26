// attn1.rs — Burnlings · Chapter 12: Attention & Transformers (SOLUTION)
//
//      Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V

use burn::backend::NdArray;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;

type Backend = NdArray;

fn attention() -> Vec<f32> {
    let device = Default::default();
    let q = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let k = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let v = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let d_k = q.dims()[1] as f64;

    let scores = q.matmul(k.transpose()).mul_scalar(1.0 / d_k.sqrt());

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
