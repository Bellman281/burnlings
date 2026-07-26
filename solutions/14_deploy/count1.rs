// count1.rs — Burnlings · Chapter 14: model footprint (SOLUTION)
//
// f16 is 2 bytes per value: f16 bytes = params * 2.

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;

type MyBackend = NdArray;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}
impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(64, 32).init(device),
            l2: LinearConfig::new(32, 10).init(device),
        }
    }
}

fn footprint() -> (usize, usize, usize, usize) {
    let device = Default::default();
    let model: Mlp<MyBackend> = Mlp::new(&device);

    let params = model.num_params();
    let f32_bytes = params * 4;
    let f16_bytes = params * 2;
    let int8_bytes = params;
    (params, f32_bytes, f16_bytes, int8_bytes)
}

fn main() {
    let (params, f32b, f16b, int8b) = footprint();
    println!("parameters   = {params}");
    println!("f32  weights = {f32b} bytes");
    println!("f16  weights = {f16b} bytes");
    println!("int8 weights = {int8b} bytes");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn footprint_at_each_precision() {
        let (params, f32b, f16b, int8b) = footprint();
        assert_eq!(params, 2410);
        assert_eq!(f32b, 9640);
        assert_eq!(f16b, 4820);
        assert_eq!(int8b, 2410);
    }
}
