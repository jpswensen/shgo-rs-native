//! Basic Rosenbrock optimization example
//! 
//! Demonstrates the simplest usage of the SHGO algorithm to find the
//! global minimum of the Rosenbrock function.

use shgo::{Shgo, ShgoOptions};
use std::time::Instant;

/// N-dimensional Rosenbrock function
/// Global minimum at x = (1, 1, ..., 1) with f(x) = 0
fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| {
            let xi = w[0];
            let xi1 = w[1];
            100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2)
        })
        .sum()
}

fn main() {
    println!("=== SHGO-RS: Rosenbrock Function Optimization ===\n");

    // 2D case
    println!("2D Rosenbrock:");
    let bounds_2d = vec![(-5.0, 5.0), (-5.0, 5.0)];
    let options_2d = ShgoOptions {
        maxiter: Some(3),
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_2d = Shgo::new(rosenbrock, bounds_2d)
        .with_options(options_2d)
        .minimize()
        .unwrap();
    let elapsed_2d = start.elapsed();

    println!("  x = {:?}", result_2d.x);
    println!("  f(x) = {:.10e}", result_2d.fun);
    println!("  Function evaluations: {}", result_2d.nfev);
    println!("  Local minimizations: {}", result_2d.nlfev);
    println!("  Iterations: {}", result_2d.nit);
    println!("  Time: {:.4} ms", elapsed_2d.as_secs_f64() * 1000.0);
    println!("  Success: {}", result_2d.success);
    println!();

    // 3D case
    println!("3D Rosenbrock:");
    let bounds_3d = vec![(-5.0, 5.0); 3];
    let options_3d = ShgoOptions {
        maxiter: Some(3),
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_3d = Shgo::new(rosenbrock, bounds_3d)
        .with_options(options_3d)
        .minimize()
        .unwrap();
    let elapsed_3d = start.elapsed();

    println!("  x = {:?}", result_3d.x);
    println!("  f(x) = {:.10e}", result_3d.fun);
    println!("  Function evaluations: {}", result_3d.nfev);
    println!("  Local minimizations: {}", result_3d.nlfev);
    println!("  Time: {:.4} ms", elapsed_3d.as_secs_f64() * 1000.0);
    println!();

    // 5D case
    println!("5D Rosenbrock:");
    let bounds_5d = vec![(-5.0, 5.0); 5];
    let options_5d = ShgoOptions {
        maxiter: Some(3),
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_5d = Shgo::new(rosenbrock, bounds_5d)
        .with_options(options_5d)
        .minimize()
        .unwrap();
    let elapsed_5d = start.elapsed();

    println!("  x = {:?}", result_5d.x);
    println!("  f(x) = {:.10e}", result_5d.fun);
    println!("  Function evaluations: {}", result_5d.nfev);
    println!("  Local minimizations: {}", result_5d.nlfev);
    println!("  Time: {:.4} ms", elapsed_5d.as_secs_f64() * 1000.0);
    println!();

    // Expected: x ≈ (1, 1, ..., 1), f(x) ≈ 0
    println!("Expected: x = (1, 1, ..., 1), f(x) = 0");
}
