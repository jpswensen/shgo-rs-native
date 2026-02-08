//! Performance benchmark: SHGO-RS timing analysis.
//!
//! Runs each test function multiple times and reports median/mean/min/max times.
//! Outputs JSON for comparison with Python.
//!
//! Build: RUSTFLAGS="-C target-cpu=native" cargo build --example performance_benchmark --release
//! Run:   ./target/release/examples/performance_benchmark

use shgo::{Shgo, ShgoOptions, SamplingMethod, LocalOptimizer};
use std::time::Instant;

// ===== Test Functions =====

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * std::f64::consts::PI;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (c * xi).cos()).sum();
    -a * (-b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + a + std::f64::consts::E
}

fn rastrigin(x: &[f64]) -> f64 {
    let a = 10.0;
    let n = x.len() as f64;
    a * n + x.iter()
        .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
}

fn eggholder(x: &[f64]) -> f64 {
    let x1 = x[0];
    let x2 = x[1];
    -(x2 + 47.0) * (x1 / 2.0 + x2 + 47.0).abs().sqrt().sin()
        - x1 * (x1 - (x2 + 47.0)).abs().sqrt().sin()
}

fn styblinski_tang(x: &[f64]) -> f64 {
    x.iter()
        .map(|&xi| xi.powi(4) - 16.0 * xi.powi(2) + 5.0 * xi)
        .sum::<f64>()
        / 2.0
}

fn levy(x: &[f64]) -> f64 {
    let w: Vec<f64> = x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();
    let n = w.len();
    let term1 = (std::f64::consts::PI * w[0]).sin().powi(2);
    let sum: f64 = w[..n - 1]
        .iter()
        .zip(w[1..].iter())
        .map(|(&wi, &_wi1)| {
            (wi - 1.0).powi(2)
                * (1.0 + 10.0 * (std::f64::consts::PI * _wi1).sin().powi(2))
        })
        .sum();
    let term3 = (w[n - 1] - 1.0).powi(2)
        * (1.0 + (2.0 * std::f64::consts::PI * w[n - 1]).sin().powi(2));
    term1 + sum + term3
}

// ===== Benchmark harness =====

struct BenchResult {
    name: String,
    fun: f64,
    nfev: usize,
    nlfev: usize,
    n_local_minima: usize,
    success: bool,
    times_us: Vec<f64>, // microseconds
}

impl BenchResult {
    fn median_us(&self) -> f64 {
        let mut sorted = self.times_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }
    fn mean_us(&self) -> f64 {
        self.times_us.iter().sum::<f64>() / self.times_us.len() as f64
    }
    fn min_us(&self) -> f64 {
        self.times_us.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn max_us(&self) -> f64 {
        self.times_us.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    fn std_us(&self) -> f64 {
        let mean = self.mean_us();
        let var = self.times_us.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / (self.times_us.len() - 1) as f64;
        var.sqrt()
    }
}

fn bench_unconstrained<F: Fn(&[f64]) -> f64 + Send + Sync + 'static + Clone>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    sampling_method: SamplingMethod,
    n: usize,
    iters: Option<usize>,
    maxiter: Option<usize>,
    n_runs: usize,
) -> BenchResult {
    let mut times_us = Vec::with_capacity(n_runs);
    let mut last_fun = 0.0;
    let mut last_nfev = 0;
    let mut last_nlfev = 0;
    let mut last_minima = 0;
    let mut last_success = false;

    for _ in 0..n_runs {
        let f = func.clone();
        let b = bounds.clone();
        let mut options = ShgoOptions {
            sampling_method,
            iters,
            maxiter,
            disp: 0,
            ..Default::default()
        };
        if n > 0 {
            options.n = n;
        }

        let start = Instant::now();
        let result = Shgo::new(f, b).with_options(options).minimize().unwrap();
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        last_fun = result.fun;
        last_nfev = result.nfev;
        last_nlfev = result.nlfev;
        last_minima = result.xl.len();
        last_success = result.success;
    }

    BenchResult {
        name: name.to_string(),
        fun: last_fun,
        nfev: last_nfev,
        nlfev: last_nlfev,
        n_local_minima: last_minima,
        success: last_success,
        times_us,
    }
}

fn bench_constrained<
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static + Clone,
    G: Fn(&[f64]) -> f64 + Send + Sync + 'static + Clone,
>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    constraints: Vec<G>,
    sampling_method: SamplingMethod,
    n: usize,
    iters: Option<usize>,
    maxiter: Option<usize>,
    n_runs: usize,
) -> BenchResult {
    let mut times_us = Vec::with_capacity(n_runs);
    let mut last_fun = 0.0;
    let mut last_nfev = 0;
    let mut last_nlfev = 0;
    let mut last_minima = 0;
    let mut last_success = false;

    for _ in 0..n_runs {
        let f = func.clone();
        let b = bounds.clone();
        let c: Vec<_> = constraints.iter().cloned().collect();
        let mut options = ShgoOptions {
            sampling_method,
            iters,
            maxiter,
            disp: 0,
            local_optimizer: LocalOptimizer::Cobyla,
            ..Default::default()
        };
        options.local_options.algorithm = LocalOptimizer::Cobyla;
        if n > 0 {
            options.n = n;
        }

        let start = Instant::now();
        let result = Shgo::with_constraints(f, b, c)
            .with_options(options)
            .minimize()
            .unwrap();
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        last_fun = result.fun;
        last_nfev = result.nfev;
        last_nlfev = result.nlfev;
        last_minima = result.xl.len();
        last_success = result.success;
    }

    BenchResult {
        name: name.to_string(),
        fun: last_fun,
        nfev: last_nfev,
        nlfev: last_nlfev,
        n_local_minima: last_minima,
        success: last_success,
        times_us,
    }
}

fn main() {
    let n_runs = 50;

    println!("======================================================================");
    println!("  RUST SHGO — Performance Benchmark ({n_runs} runs per test)");
    println!("======================================================================");

    // Warmup: run one optimization to load libraries, warm caches
    let _ = Shgo::new(sphere, vec![(-5.0, 5.0); 2])
        .minimize()
        .unwrap();

    let mut results: Vec<BenchResult> = Vec::new();

    // --- 1. Sphere 2D simplicial ---
    let r = bench_unconstrained(
        "sphere_2d_simp", sphere,
        vec![(-5.0, 5.0); 2],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 2. Sphere 2D sobol ---
    let r = bench_unconstrained(
        "sphere_2d_sobol", sphere,
        vec![(-5.0, 5.0); 2],
        SamplingMethod::Sobol, 64, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 3. Rosenbrock 2D simplicial ---
    let r = bench_unconstrained(
        "rosenbrock_2d_simp", rosenbrock,
        vec![(-5.0, 5.0); 2],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 4. Rosenbrock 2D sobol ---
    let r = bench_unconstrained(
        "rosenbrock_2d_sobol", rosenbrock,
        vec![(-5.0, 5.0); 2],
        SamplingMethod::Sobol, 128, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 5. Ackley 2D simplicial ---
    let r = bench_unconstrained(
        "ackley_2d_simp", ackley,
        vec![(-5.0, 5.0); 2],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 6. Rastrigin 2D simplicial ---
    let r = bench_unconstrained(
        "rastrigin_2d_simp", rastrigin,
        vec![(-5.12, 5.12); 2],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 7. Rastrigin 2D sobol ---
    let r = bench_unconstrained(
        "rastrigin_2d_sobol", rastrigin,
        vec![(-5.12, 5.12); 2],
        SamplingMethod::Sobol, 128, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 8. Eggholder 2D simplicial (iters=3) ---
    let r = bench_unconstrained(
        "eggholder_2d_simp", eggholder,
        vec![(-512.0, 512.0); 2],
        SamplingMethod::Simplicial, 0, Some(3), None, n_runs,
    );
    results.push(r);

    // --- 9. Eggholder 2D sobol ---
    let r = bench_unconstrained(
        "eggholder_2d_sobol", eggholder,
        vec![(-512.0, 512.0); 2],
        SamplingMethod::Sobol, 256, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 10. Styblinski-Tang 2D simplicial ---
    let r = bench_unconstrained(
        "styblinski_2d_simp", styblinski_tang,
        vec![(-5.0, 5.0); 2],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 11. Levy 2D simplicial ---
    let r = bench_unconstrained(
        "levy_2d_simp", levy,
        vec![(-10.0, 10.0); 2],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 12. Sphere 5D simplicial ---
    let r = bench_unconstrained(
        "sphere_5d_simp", sphere,
        vec![(-5.0, 5.0); 5],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 13. Rosenbrock 3D simplicial ---
    let r = bench_unconstrained(
        "rosenbrock_3d_simp", rosenbrock,
        vec![(-5.0, 5.0); 3],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 14. Sphere 2D constrained (COBYLA) ---
    let constraint = |x: &[f64]| x[0] + x[1] - 1.0;
    let r = bench_constrained(
        "sphere_2d_constrained", sphere,
        vec![(0.0, 2.0), (0.0, 2.0)],
        vec![constraint],
        SamplingMethod::Simplicial, 0, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 15. Rastrigin 3D sobol ---
    let r = bench_unconstrained(
        "rastrigin_3d_sobol", rastrigin,
        vec![(-5.12, 5.12); 3],
        SamplingMethod::Sobol, 128, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 16. Rosenbrock 5D sobol (harder) ---
    let r = bench_unconstrained(
        "rosenbrock_5d_sobol", rosenbrock,
        vec![(-5.0, 5.0); 5],
        SamplingMethod::Sobol, 256, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 17. Ackley 5D sobol ---
    let r = bench_unconstrained(
        "ackley_5d_sobol", ackley,
        vec![(-5.0, 5.0); 5],
        SamplingMethod::Sobol, 256, Some(1), None, n_runs,
    );
    results.push(r);

    // --- 18. Rastrigin 5D sobol ---
    let r = bench_unconstrained(
        "rastrigin_5d_sobol", rastrigin,
        vec![(-5.12, 5.12); 5],
        SamplingMethod::Sobol, 256, Some(1), None, n_runs,
    );
    results.push(r);

    // ===== Print results =====
    println!("\n{:─<78}", "");
    println!(
        "  {:<26} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Test", "median(μs)", "mean(μs)", "min(μs)", "std(μs)", "f(x)"
    );
    println!("{:─<78}", "");
    for r in &results {
        println!(
            "  {:<26} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>8.2e}",
            r.name,
            r.median_us(),
            r.mean_us(),
            r.min_us(),
            r.std_us(),
            r.fun,
        );
    }
    println!();

    // ===== Write JSON =====
    // Manual JSON since serde is dev-dependency
    let mut json = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        json.push_str(&format!(
            "  {{\n    \"name\": \"{}\",\n    \"fun\": {:.16e},\n    \"nfev\": {},\n    \"nlfev\": {},\n    \"n_local_minima\": {},\n    \"success\": {},\n    \"n_runs\": {},\n    \"median_us\": {:.2},\n    \"mean_us\": {:.2},\n    \"min_us\": {:.2},\n    \"max_us\": {:.2},\n    \"std_us\": {:.2},\n    \"all_times_us\": [{}]\n  }}",
            r.name, r.fun, r.nfev, r.nlfev, r.n_local_minima, r.success,
            r.times_us.len(),
            r.median_us(), r.mean_us(), r.min_us(), r.max_us(), r.std_us(),
            r.times_us.iter().map(|t| format!("{:.2}", t)).collect::<Vec<_>>().join(", "),
        ));
        if i < results.len() - 1 {
            json.push_str(",\n");
        }
        json.push('\n');
    }
    json.push_str("]\n");

    let out_path = std::path::Path::new("rust_perf_results.json");
    std::fs::write(out_path, &json).expect("Failed to write JSON");
    println!("  Results written to {}", out_path.display());
}
