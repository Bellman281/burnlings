// opt1.rs — Burnlings · Chapter 9: Training
//
// Swapping optimizers in Burn changes only the config line. For SGD-with-
// momentum the momentum must be wrapped in `Some`, because `with_momentum` takes
// an `Option<MomentumConfig>` (passing `None` means plain SGD). Handing it a bare
// `MomentumConfig` doesn't type-check.
//
// TODO: Wrap the momentum config in `Some(...)`.
//       (Compare book example chapter09/e02 — choosing an optimizer.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::nn::Linear;
use burn::optim::momentum::MomentumConfig;
use burn::optim::SgdConfig;

type B = Autodiff<NdArray>;

fn build_sgd_with_momentum() {
    // ⬇️ `with_momentum` wants an Option — a bare MomentumConfig won't compile
    let _optim = SgdConfig::new()
        .with_momentum(MomentumConfig::new())
        .init::<B, Linear<B>>();
}

fn main() {
    build_sgd_with_momentum();
    println!("SGD-with-momentum optimizer built");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builds() {
        build_sgd_with_momentum();
    }
}
