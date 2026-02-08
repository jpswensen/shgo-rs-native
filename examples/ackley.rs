//! Ackley function optimization
//! 
//! A widely used test function for optimization algorithms.

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

/// N-dimensional Ackley function
/// Global minimum at x = (0, 0, ..., 0) with f(x) = 0
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * std::f64::consts::PI;

    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (c * xi).cos()).sum();

    -a * (-b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + a + std::f64::consts::E
}

fn main() {
    println!("=== SHGO-RS: Ackley Function Optimization ===\n");

    for dim in [2, 3, 5] {
        println!("{}D Ackley:", dim);
        let bounds = vec![(-5.0, 5.0); dim];

        let options = ShgoOptions {
            maxiter: Some(3),
            sampling_method: SamplingMethod::Simplicial,
            disp: 0,
            ..Default::default()
        };

        let start = Instant::now();
        let result = Shgo::new(ackley, bounds)
            .with_options(options)
            .minimize()
            .unwrap();
        let elapsed = start.elapsed();

        println!("  x = {:?}", result.x);
        println!("  f(x) = {:.10e}", result.fun);
        println!("  Function evaluations: {}", result.nfev);
        println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
        println!();
    }

    println!("Expected: x = (0, 0, ..., 0), f(x) = 0");
}
