// attn2.rs — Burnlings · Chapter 12: Attention & Transformers (SOLUTION)
//
// Causal masking: add -1e9 to every future score before the softmax.

use burn::backend::NdArray;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;

type Backend = NdArray;

fn masked_attention() -> Vec<f32> {
    let device = Default::default();
    let q = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let k = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let v = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let d_k = q.dims()[1] as f64;
    let scale = 1.0 / d_k.sqrt();

    let scores = q.matmul(k.transpose()).mul_scalar(scale);

    let mask = Tensor::<Backend, 2>::from_floats([[0.0, -1.0e9], [0.0, 0.0]], &device);
    let masked = scores + mask;

    let weights = softmax(masked, 1);
    let out = weights.matmul(v);
    out.into_data().to_vec().unwrap()
}

fn main() {
    println!("masked attention output = {:?}", masked_attention());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_mask_hides_the_future() {
        let out = masked_attention();
        let expect = [1.0, 2.0, 2.339523, 3.339523];
        for (o, e) in out.iter().zip(expect.iter()) {
            assert!((o - e).abs() < 1e-3, "got {o}, want {e}");
        }
    }
}
