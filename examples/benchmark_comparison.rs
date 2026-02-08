//! Comprehensive benchmark comparing different test functions and settings
//! 
//! This outputs results in JSON format for easy comparison with Python scipy.optimize.shgo

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

/// Sphere function: f(x) = sum(x_i^2)
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Rosenbrock function
fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

/// Rastrigin function
fn rastrigin(x: &[f64]) -> f64 {
    let a = 10.0;
    let n = x.len() as f64;
    a * n + x.iter()
        .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
}

/// Ackley function
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * std::f64::consts::PI;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (c * xi).cos()).sum();
    -a * (-b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + a + std::f64::consts::E
}

/// Eggholder function (2D only)
fn eggholder(x: &[f64]) -> f64 {
    let x1 = x[0];
    let x2 = x[1];
    -(x2 + 47.0) * (x1 / 2.0 + x2 + 47.0).abs().sqrt().sin()
        - x1 * (x1 - (x2 + 47.0)).abs().sqrt().sin()
}

struct BenchmarkResult {
    function: String,
    dim: usize,
    sampling: String,
    x: Vec<f64>,
    fun: f64,
    nfev: usize,
    nlfev: usize,
    nit: usize,
    time_ms: f64,
    success: bool,
}

fn run_benchmark<F: Fn(&[f64]) -> f64 + Send + Sync + 'static>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    sampling: SamplingMethod,
    maxiter: usize,
    n: usize,
) -> BenchmarkResult {
    let dim = bounds.len();
    let options = ShgoOptions {
        maxiter: Some(maxiter),
        n,
        sampling_method: sampling,
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(func, bounds)
        .with_options(options)
        .minimize()
        .unwrap();
    let elapsed = start.elapsed();

    BenchmarkResult {
        function: name.to_string(),
        dim,
        sampling: format!("{:?}", sampling),
        x: result.x.clone(),
        fun: result.fun,
        nfev: result.nfev,
        nlfev: result.nlfev,
        nit: result.nit,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        success: result.success,
    }
}

fn print_result(r: &BenchmarkResult) {
    println!("{},{},{},{:.10e},{},{},{},{:.4},{}", 
        r.function, r.dim, r.sampling, r.fun, r.nfev, r.nlfev, r.nit, r.time_ms, r.success);
}

fn main() {
    println!("=== SHGO-RS Comprehensive Benchmark ===\n");
    
    // CSV header
    println!("function,dim,sampling,fun,nfev,nlfev,nit,time_ms,success");
    
    // Sphere tests
    for dim in [2, 3, 5] {
        let bounds = vec![(-5.0, 5.0); dim];
        let r = run_benchmark("sphere", sphere, bounds.clone(), SamplingMethod::Simplicial, 3, 128);
        print_result(&r);
        let r = run_benchmark("sphere", sphere, bounds, SamplingMethod::Sobol, 3, 64);
        print_result(&r);
    }

    // Rosenbrock tests
    for dim in [2, 3, 5] {
        let bounds = vec![(-5.0, 5.0); dim];
        let r = run_benchmark("rosenbrock", rosenbrock, bounds.clone(), SamplingMethod::Simplicial, 3, 128);
        print_result(&r);
        let r = run_benchmark("rosenbrock", rosenbrock, bounds, SamplingMethod::Sobol, 3, 64);
        print_result(&r);
    }

    // Rastrigin tests
    for dim in [2, 3] {
        let bounds = vec![(-5.12, 5.12); dim];
        let r = run_benchmark("rastrigin", rastrigin, bounds.clone(), SamplingMethod::Simplicial, 3, 128);
        print_result(&r);
        let r = run_benchmark("rastrigin", rastrigin, bounds, SamplingMethod::Sobol, 3, 64);
        print_result(&r);
    }

    // Ackley tests
    for dim in [2, 3] {
        let bounds = vec![(-5.0, 5.0); dim];
        let r = run_benchmark("ackley", ackley, bounds.clone(), SamplingMethod::Simplicial, 3, 128);
        print_result(&r);
        let r = run_benchmark("ackley", ackley, bounds, SamplingMethod::Sobol, 3, 64);
        print_result(&r);
    }

    // Eggholder (2D only)
    let bounds = vec![(-512.0, 512.0), (-512.0, 512.0)];
    let r = run_benchmark("eggholder", eggholder, bounds.clone(), SamplingMethod::Simplicial, 5, 128);
    print_result(&r);
    let r = run_benchmark("eggholder", eggholder, bounds, SamplingMethod::Sobol, 3, 256);
    print_result(&r);

    println!("\n=== Expected Global Minima ===");
    println!("sphere:     f(0, 0, ...) = 0");
    println!("rosenbrock: f(1, 1, ...) = 0");
    println!("rastrigin:  f(0, 0, ...) = 0");
    println!("ackley:     f(0, 0, ...) = 0");
    println!("eggholder:  f(512, 404.2319) ≈ -959.6407");
}
