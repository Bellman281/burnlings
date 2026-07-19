// tensors3.rs — Burnlings · Chapter 1: Tensors
//
// Integer tensors are a DIFFERENT kind from float tensors. The kind is the
// third type parameter of `Tensor`: `Tensor<B, D, Int>`. Leave it off and you
// get a float tensor, which is not what `ids` promises to return.
//
// TODO: Make the body build an Int tensor so it matches the return type.
//       You'll need the `Int` kind in the type annotation.
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn ids() -> Tensor<Backend, 1, Int> {
    let device = Default::default();
    // ⬇️ this builds a FLOAT tensor (no `Int` kind)
    Tensor::<Backend, 1>::from_data(TensorData::from([10, 20, 30]), &device)
}

fn main() {
    let t = ids();
    println!("ids = {t}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn is_int_rank1_len3() {
        let t = ids();
        assert_eq!(t.dims(), [3]);
        let d = t.into_data();
        let v: &[i64] = d.as_slice().unwrap();
        assert_eq!(v, [10, 20, 30]);
    }
}
