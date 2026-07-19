// tensors6.rs — SOLUTION · Chapter 1: Tensors
// A Rust array must be homogeneous. To pack mixed-typed struct fields into one
// float tensor, cast each field to f32 first.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

type Backend = NdArray;

struct Reading {
    sensor_id: i32,
    count: i16,
    voltage: f32,
}

fn to_tensor(r: &Reading) -> Tensor<Backend, 1> {
    let device = Default::default();
    let data = TensorData::from([r.sensor_id as f32, r.count as f32, r.voltage]);
    Tensor::<Backend, 1>::from_data(data, &device)
}

fn main() {
    let r = Reading { sensor_id: 7, count: 42, voltage: 3.3 };
    println!("t = {}", to_tensor(&r));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packs_three_fields() {
        let r = Reading { sensor_id: 7, count: 42, voltage: 3.3 };
        let t = to_tensor(&r);
        assert_eq!(t.dims(), [3]);
        let d = t.into_data();
        let v: &[f32] = d.as_slice().unwrap();
        assert_eq!(v, [7.0, 42.0, 3.3]);
    }
}
