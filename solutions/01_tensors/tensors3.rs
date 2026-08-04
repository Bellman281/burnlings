// tensors3.rs — SOLUTION · Chapter 1: Tensors
// Integer tensors are a DIFFERENT kind from float tensors. The kind is the
// third type parameter: `Tensor<B, D, Int>`.

use burn::backend::Flex;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = Flex;

fn ids() -> Tensor<Backend, 1, Int> {
    let device = Default::default();
    Tensor::<Backend, 1, Int>::from_data(TensorData::from([10, 20, 30]), &device)
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
        // `iter::<i64>()` converts on read, so this assertion holds on any
        // backend regardless of its native int element type (Flex: i32).
        let d = t.into_data();
        let v: Vec<i64> = d.iter::<i64>().collect();
        assert_eq!(v, [10, 20, 30]);
    }
}
