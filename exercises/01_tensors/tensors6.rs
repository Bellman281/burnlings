// tensors6.rs — Burnlings · Chapter 1: Tensors
//
// A Rust array must be homogeneous — every element the same type. A struct's
// fields often are not (here: i32, i16, f32). To pack them into one float
// tensor, cast each field to f32 first.
//
// The array below mixes i32, i16 and f32, so it will not compile.
//
// TODO: Cast the integer fields so the array is all f32.
//       (Compare book example ch1_06_from_custom_type.rs.)
//
// I AM NOT DONE

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
    // ⬇️ mixed types in one array: i32, i16, f32
    let data = TensorData::from([r.sensor_id, r.count, r.voltage]);
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
