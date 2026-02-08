//! Eggholder function optimization
//! 
//! A challenging 2D test function with many local minima.

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

/// Eggholder function (2D only)
/// Global minimum at x ≈ (512, 404.2319) with f(x) ≈ -959.6407
fn eggholder(x: &[f64]) -> f64 {
    let x1 = x[0];
    let x2 = x[1];
    -(x2 + 47.0) * (x1 / 2.0 + x2 + 47.0).abs().sqrt().sin()
        - x1 * (x1 - (x2 + 47.0)).abs().sqrt().sin()
}

fn main() {
    println!("=== SHGO-RS: Eggholder Function Optimization ===\n");

    let bounds = vec![(-512.0, 512.0), (-512.0, 512.0)];

    // Simplicial sampling with more iterations
    println!("Simplicial sampling:");
    let options_simp = ShgoOptions {
        maxiter: Some(5),
        sampling_method: SamplingMethod::Simplicial,
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_simp = Shgo::new(eggholder, bounds.clone())
        .with_options(options_simp)
        .minimize()
        .unwrap();
    let elapsed_simp = start.elapsed();

    println!("  x = {:?}", result_simp.x);
    println!("  f(x) = {:.6}", result_simp.fun);
    println!("  Function evaluations: {}", result_simp.nfev);
    println!("  Local minima found: {}", result_simp.xl.len());
    println!("  Time: {:.4} ms", elapsed_simp.as_secs_f64() * 1000.0);
    println!();

    // Sobol sampling with many points
    println!("Sobol sampling (n=256):");
    let options_sobol = ShgoOptions {
        n: 256,
        maxiter: Some(3),
        sampling_method: SamplingMethod::Sobol,
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_sobol = Shgo::new(eggholder, bounds)
        .with_options(options_sobol)
        .minimize()
        .unwrap();
    let elapsed_sobol = start.elapsed();

    println!("  x = {:?}", result_sobol.x);
    println!("  f(x) = {:.6}", result_sobol.fun);
    println!("  Function evaluations: {}", result_sobol.nfev);
    println!("  Local minima found: {}", result_sobol.xl.len());
    println!("  Time: {:.4} ms", elapsed_sobol.as_secs_f64() * 1000.0);
    println!();

    println!("Expected: x ≈ (512, 404.2319), f(x) ≈ -959.6407");
}
