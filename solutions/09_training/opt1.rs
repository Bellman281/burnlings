// opt1.rs — SOLUTION · Chapter 9: Training
// Swapping optimizers in Burn changes only the config line. SGD-with-momentum is
// `SgdConfig::new().with_momentum(Some(MomentumConfig::new()))` — note the
// momentum is wrapped in `Some`, because `with_momentum` takes an `Option`.

use burn::backend::{Autodiff, NdArray};
use burn::nn::Linear;
use burn::optim::momentum::MomentumConfig;
use burn::optim::SgdConfig;

type B = Autodiff<NdArray>;

fn build_sgd_with_momentum() {
    let _optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new()))
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
