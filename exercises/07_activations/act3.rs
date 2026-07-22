// act3.rs — Burnlings · Chapter 7: Activations
//
// Softmax turns logits into a probability distribution that sums to 1. For a
// [batch, classes] tensor you must normalise over the CLASS axis (dim 1).
// Normalising over the batch axis (dim 0) compiles fine but is wrong — with a
// single row it just yields [1, 1, 1], which doesn't match a real distribution.
//
// TODO: Softmax over the class dimension (dim 1), not the batch dimension.
//       (Compare book example chapter07/e03_softmax.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn probs() -> Vec<f32> {
    let device = Default::default();
    // one row: [batch=1, classes=3]
    let logits = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    // ⬇️ dim 0 normalises over the batch — wrong axis
    activation::softmax(logits, 0).into_data().to_vec().unwrap()
}

fn main() {
    println!("softmax = {:?}", probs());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn is_a_distribution_over_classes() {
        let v = probs();
        let expected = [0.09003057, 0.24472848, 0.66524094];
        for (g, w) in v.iter().zip(expected.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }
}
