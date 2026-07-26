// attn2.rs — Burnlings · Chapter 12: Attention & Transformers
//
// Causal (masked) self-attention. In a language model, token i must NOT attend
// to tokens that come after it (positions j > i). We enforce that with an
// additive mask: add a large NEGATIVE number to every "future" score BEFORE the
// softmax, so those weights collapse to zero.
//
// Get the sign wrong and you mask the wrong thing — a large POSITIVE number
// makes the future position dominate, so the token attends only ahead.
//
// TODO: The forbidden (future) score must become a large NEGATIVE number so its
//       softmax weight → 0. Fix the sign of the mask entry (`-1.0e9`).
//       (Compare book example chapter12/e02_causal_masking.)
//
// I AM NOT DONE

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

    // ⬇️ wrong sign: +1e9 makes the FUTURE position win; it must be -1e9 so the
    //    future is masked OUT and row 0 can only see position 0.
    let mask = Tensor::<Backend, 2>::from_floats([[0.0, 1.0e9], [0.0, 0.0]], &device);
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
        // Row 0 can only see position 0, so it copies V's first row exactly: [1, 2].
        let out = masked_attention();
        let expect = [1.0, 2.0, 2.339523, 3.339523];
        for (o, e) in out.iter().zip(expect.iter()) {
            assert!((o - e).abs() < 1e-3, "got {o}, want {e}");
        }
    }
}
