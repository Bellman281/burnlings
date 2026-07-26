// quant1.rs — Burnlings · Chapter 14: int8 quantisation (SOLUTION)
//
// q = round(w / scale) as i8.

fn quantize() -> Vec<i8> {
    let weights: [f32; 5] = [0.5, -1.2, 0.03, 0.9, -0.4];

    let max_abs = weights.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    let scale = max_abs / 127.0;

    weights.iter().map(|&w| (w / scale).round() as i8).collect()
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
        assert_eq!(quantize(), vec![53, -127, 3, 95, -42]);
    }
}
