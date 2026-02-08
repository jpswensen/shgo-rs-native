//! Rastrigin function optimization example
//! 
//! The Rastrigin function is a non-convex function with many local minima,
//! making it a challenging test case for global optimization algorithms.

use shgo::{Shgo, ShgoOptions, SamplingMethod, LocalOptimizer};
use std::time::Instant;

/// N-dimensional Rastrigin function
/// Global minimum at x = (0, 0, ..., 0) with f(x) = 0
fn rastrigin(x: &[f64]) -> f64 {
    let a = 10.0;
    let n = x.len() as f64;
    a * n + x.iter()
        .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
}

/// Rastrigin with configurable 'a' parameter
fn rastrigin_parametric(x: &[f64], a: f64) -> f64 {
    let n = x.len() as f64;
    a * n + x.iter()
        .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
}

fn main() {
    println!("=== SHGO-RS: Rastrigin Function Optimization ===\n");

    // 2D case with Simplicial sampling
    println!("2D Rastrigin (Simplicial sampling):");
    let bounds_2d = vec![(-5.12, 5.12), (-5.12, 5.12)];
    let options = ShgoOptions {
        maxiter: Some(5),
        sampling_method: SamplingMethod::Simplicial,
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(rastrigin, bounds_2d.clone())
        .with_options(options)
        .minimize()
        .unwrap();
    let elapsed = start.elapsed();

    println!("  x = {:?}", result.x);
    println!("  f(x) = {:.10e}", result.fun);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Local minimizations: {}", result.nlfev);
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  All local minima found: {}", result.xl.len());
    println!();

    // 2D case with Sobol sampling
    println!("2D Rastrigin (Sobol sampling):");
    let options_sobol = ShgoOptions {
        n: 128,
        maxiter: Some(3),
        sampling_method: SamplingMethod::Sobol,
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_sobol = Shgo::new(rastrigin, bounds_2d)
        .with_options(options_sobol)
        .minimize()
        .unwrap();
    let elapsed_sobol = start.elapsed();

    println!("  x = {:?}", result_sobol.x);
    println!("  f(x) = {:.10e}", result_sobol.fun);
    println!("  Function evaluations: {}", result_sobol.nfev);
    println!("  Local minimizations: {}", result_sobol.nlfev);
    println!("  Time: {:.4} ms", elapsed_sobol.as_secs_f64() * 1000.0);
    println!();

    // 3D case with parametric closure
    println!("3D Rastrigin (parametric, a=10.0):");
    let bounds_3d = vec![(-5.12, 5.12); 3];
    let a = 10.0;
    let rastrigin_partial = move |x: &[f64]| rastrigin_parametric(x, a);

    let options_3d = ShgoOptions {
        maxiter: Some(3),
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_3d = Shgo::new(rastrigin_partial, bounds_3d)
        .with_options(options_3d)
        .minimize()
        .unwrap();
    let elapsed_3d = start.elapsed();

    println!("  x = {:?}", result_3d.x);
    println!("  f(x) = {:.10e}", result_3d.fun);
    println!("  Function evaluations: {}", result_3d.nfev);
    println!("  Time: {:.4} ms", elapsed_3d.as_secs_f64() * 1000.0);
    println!();

    // Compare different local optimizers
    println!("2D Rastrigin - Comparing local optimizers:");
    let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
    
    for optimizer in [LocalOptimizer::Bobyqa, LocalOptimizer::NelderMead, LocalOptimizer::Cobyla] {
        let options = ShgoOptions {
            maxiter: Some(3),
            local_optimizer: optimizer,
            disp: 0,
            ..Default::default()
        };

        let start = Instant::now();
        let result = Shgo::new(rastrigin, bounds.clone())
            .with_options(options)
            .minimize()
            .unwrap();
        let elapsed = start.elapsed();

        println!("  {:?}: f(x) = {:.6e}, time = {:.4} ms", 
            optimizer, result.fun, elapsed.as_secs_f64() * 1000.0);
    }
    println!();

    println!("Expected: x = (0, 0, ..., 0), f(x) = 0");
}
