// quant1.rs — Burnlings · Chapter 14: int8 quantisation
//
// Symmetric int8 quantisation, the standard scheme for shrinking weights ~4x:
//      scale = max(|w|) / 127
//      q     = round(w / scale)   -> stored as i8, one byte each
//      w'    = q * scale          -> dequantise at inference time
//
// The ROUND matters: casting f32 to i8 truncates toward zero, so a value that
// should round UP to the next code is left one short. This is pure arithmetic
// (no framework) so the mechanism is fully visible.
//
// TODO: Round to the nearest integer before casting: `(w / scale).round() as i8`.
//       (Compare book example chapter14/e03_int8_quantisation_by_hand.)
//
// I AM NOT DONE

fn quantize() -> Vec<i8> {
    let weights: [f32; 5] = [0.5, -1.2, 0.03, 0.9, -0.4];

    // scale from the largest-magnitude weight.
    let max_abs = weights.iter().fold(0.0f32, |m, &w| m.max(w.abs())); // 1.2
    let scale = max_abs / 127.0;

    // ⬇️ `as i8` truncates toward zero — quantisation must ROUND to nearest
    weights.iter().map(|&w| (w / scale) as i8).collect()
}

fn main() {
    let q = quantize();
    println!("int8 codes = {q:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quantises_by_rounding() {
        // 0.5 / (1.2/127) = 52.9 -> rounds to 53, not 52.
        assert_eq!(quantize(), vec![53, -127, 3, 95, -42]);
    }
}
