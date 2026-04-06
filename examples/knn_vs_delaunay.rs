//! Compare all four connectivity methods for Sobol mode:
//! Delaunay, k-NN, HNSW, and ScaNN.
//!
//! Run: cargo run --release --example knn_vs_delaunay

use std::time::Instant;

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
}

struct BenchCase {
    name: &'static str,
    dim: usize,
    n: usize,
    func: fn(&[f64]) -> f64,
    bounds_range: (f64, f64),
}

struct MethodResult {
    median_ms: f64,
    best_fun: f64,
}

fn bench_method(
    method: shgo::ConnectivityMethod,
    case: &BenchCase,
    repeats: usize,
) -> MethodResult {
    use shgo::{Shgo, ShgoOptions, SamplingMethod};

    let bounds = vec![case.bounds_range; case.dim];
    let mut times = Vec::new();
    let mut best_fun = f64::MAX;

    for _ in 0..repeats {
        let t = Instant::now();
        let result = Shgo::new(case.func, bounds.clone())
            .with_options(ShgoOptions {
                sampling_method: SamplingMethod::Sobol,
                connectivity_method: method,
                n: case.n,
                iters: Some(1),
                disp: 0,
                ..Default::default()
            })
            .minimize();
        let elapsed = t.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
        if let Ok(r) = result {
            best_fun = best_fun.min(r.fun);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    MethodResult {
        median_ms: times[repeats / 2],
        best_fun,
    }
}

fn main() {
    use shgo::ConnectivityMethod;

    let cases = vec![
        BenchCase { name: "Rastrigin 2D n=128", dim: 2, n: 128, func: rastrigin, bounds_range: (-5.12, 5.12) },
        BenchCase { name: "Rastrigin 3D n=128", dim: 3, n: 128, func: rastrigin, bounds_range: (-5.12, 5.12) },
        BenchCase { name: "Ackley 5D n=256",    dim: 5, n: 256, func: ackley,    bounds_range: (-5.0, 5.0) },
        BenchCase { name: "Rastrigin 5D n=512", dim: 5, n: 512, func: rastrigin, bounds_range: (-5.12, 5.12) },
        BenchCase { name: "Ackley 7D n=256",    dim: 7, n: 256, func: ackley,    bounds_range: (-5.0, 5.0) },
        BenchCase { name: "Rastrigin 7D n=512", dim: 7, n: 512, func: rastrigin, bounds_range: (-5.12, 5.12) },
    ];

    let methods: Vec<(&str, ConnectivityMethod)> = vec![
        ("Delaunay", ConnectivityMethod::Delaunay),
        ("k-NN",     ConnectivityMethod::KNearestNeighbors),
        ("HNSW",     ConnectivityMethod::HNSW),
        ("ScaNN",    ConnectivityMethod::ScaNN),
    ];

    let repeats = 3;

    println!("\n=== Connectivity Method Comparison (4 methods) ===\n");

    // Header
    print!("{:<25}", "Case");
    for (name, _) in &methods {
        print!(" {:>10}", format!("{}(ms)", name));
    }
    print!("  {:>10}", "Best");
    println!();
    println!("{}", "-".repeat(25 + methods.len() * 11 + 12));

    for case in &cases {
        let results: Vec<MethodResult> = methods.iter()
            .map(|(_, m)| bench_method(*m, case, repeats))
            .collect();

        // Find fastest
        let min_time = results.iter().map(|r| r.median_ms).fold(f64::INFINITY, f64::min);
        let best_idx = results.iter().position(|r| (r.median_ms - min_time).abs() < 0.01).unwrap_or(0);

        print!("{:<25}", case.name);
        for r in &results {
            print!(" {:>10.1}", r.median_ms);
        }
        print!("  {:>10}", methods[best_idx].0);
        println!();
    }

    // Solution quality comparison
    println!("\n=== Solution Quality (best f(x) found) ===\n");
    print!("{:<25}", "Case");
    for (name, _) in &methods {
        print!(" {:>14}", name);
    }
    println!();
    println!("{}", "-".repeat(25 + methods.len() * 15));

    for case in &cases {
        let results: Vec<MethodResult> = methods.iter()
            .map(|(_, m)| bench_method(*m, case, repeats))
            .collect();

        print!("{:<25}", case.name);
        for r in &results {
            print!(" {:>14.4e}", r.best_fun);
        }
        println!();
    }
    println!();
}
