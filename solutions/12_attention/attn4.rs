// attn4.rs — Burnlings · Chapter 12: Attention & Transformers (SOLUTION)
//
// A transformer block: both residuals wrapped in LayerNorm.

use burn::backend::Flex;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend as _;
use burn::tensor::Tensor;

type Backend = Flex;

fn layer_norm(x: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = (centered.clone() * centered.clone()).mean_dim(1);
    centered / var.add_scalar(1e-5).sqrt()
}

fn block() -> (Vec<usize>, Vec<f32>) {
    let device = Default::default();
    Backend::seed(&device, 7);

    let d_model = 4;
    let d_ff = 8;

    let x = Tensor::<Backend, 2>::from_floats(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
        &device,
    );

    let wq: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wk: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wv: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let wo: Linear<Backend> = LinearConfig::new(d_model, d_model).init(&device);
    let ff1: Linear<Backend> = LinearConfig::new(d_model, d_ff).init(&device);
    let ff2: Linear<Backend> = LinearConfig::new(d_ff, d_model).init(&device);

    let q = wq.forward(x.clone());
    let k = wk.forward(x.clone());
    let v = wv.forward(x.clone());
    let scale = 1.0 / (d_model as f64).sqrt();
    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let weights = softmax(scores, 1);
    let attn = wo.forward(weights.matmul(v));
    let x = layer_norm(x + attn);

    let ff = ff2.forward(relu(ff1.forward(x.clone())));
    let out = layer_norm(x + ff);

    let means = out.clone().mean_dim(1).into_data().to_vec().unwrap();
    (out.dims().to_vec(), means)
}

fn main() {
    let (dims, means) = block();
    println!("output dims = {dims:?}");
    println!("row means (should be ~0) = {means:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_output_is_normalised() {
        let (dims, means) = block();
        assert_eq!(dims, vec![3, 4]);
        for m in &means {
            assert!(m.abs() < 1e-4, "row mean {m} is not ~0");
        }
    }
}
