//! Sphere function benchmark
//! 
//! Simple convex function for basic performance testing.

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

/// N-dimensional Sphere function
/// Global minimum at x = (0, 0, ..., 0) with f(x) = 0
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn main() {
    println!("=== SHGO-RS: Sphere Function Benchmark ===\n");

    for dim in [2, 3, 5, 10] {
        println!("{}D Sphere:", dim);
        let bounds = vec![(-5.0, 5.0); dim];

        // Simplicial
        let options_simp = ShgoOptions {
            maxiter: Some(3),
            sampling_method: SamplingMethod::Simplicial,
            disp: 0,
            ..Default::default()
        };

        let start = Instant::now();
        let result_simp = Shgo::new(sphere, bounds.clone())
            .with_options(options_simp)
            .minimize()
            .unwrap();
        let elapsed_simp = start.elapsed();

        // Sobol
        let options_sobol = ShgoOptions {
            n: 64,
            maxiter: Some(3),
            sampling_method: SamplingMethod::Sobol,
            disp: 0,
            ..Default::default()
        };

        let start = Instant::now();
        let result_sobol = Shgo::new(sphere, bounds)
            .with_options(options_sobol)
            .minimize()
            .unwrap();
        let elapsed_sobol = start.elapsed();

        println!("  Simplicial: f(x) = {:.6e}, nfev = {}, time = {:.4} ms",
            result_simp.fun, result_simp.nfev, elapsed_simp.as_secs_f64() * 1000.0);
        println!("  Sobol:      f(x) = {:.6e}, nfev = {}, time = {:.4} ms",
            result_sobol.fun, result_sobol.nfev, elapsed_sobol.as_secs_f64() * 1000.0);
        println!();
    }

    println!("Expected: x = (0, 0, ..., 0), f(x) = 0");
}
