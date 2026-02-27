//! Parallel performance benchmark: compares single-threaded vs multi-threaded Rust SHGO.
//!
//! Tests longer-running optimizations where parallelization of local
//! minimization actually matters. Each test is run with workers=1 (serial)
//! and workers=None (all CPUs via rayon), and results are written to JSON.
//!
//! Build: RUSTFLAGS="-C target-cpu=native" cargo build --example parallel_benchmark --release
//! Run:   ./target/release/examples/parallel_benchmark

use shgo::{Shgo, ShgoOptions, SamplingMethod, LocalOptimizer};
use std::sync::Arc;
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

fn levy(x: &[f64]) -> f64 {
    let w: Vec<f64> = x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();
    let n = w.len();
    let term1 = (std::f64::consts::PI * w[0]).sin().powi(2);
    let sum: f64 = w[..n - 1]
        .iter()
        .zip(w[1..].iter())
        .map(|(&wi, &wi1)| {
            (wi - 1.0).powi(2) * (1.0 + 10.0 * (std::f64::consts::PI * wi1).sin().powi(2))
        })
        .sum();
    let term3 = (w[n - 1] - 1.0).powi(2)
        * (1.0 + (2.0 * std::f64::consts::PI * w[n - 1]).sin().powi(2));
    term1 + sum + term3
}

/// Expensive wrapper: adds a busy-loop (~100μs) to simulate a costly evaluation.
fn make_expensive<F: Fn(&[f64]) -> f64>(func: F, work_us: u64) -> impl Fn(&[f64]) -> f64 {
    move |x: &[f64]| {
        let val = func(x);
        // Busy-wait to simulate expensive computation
        let target = std::time::Duration::from_micros(work_us);
        let start = Instant::now();
        while start.elapsed() < target {
            std::hint::spin_loop();
        }
        val
    }
}

// ===== Benchmark harness =====

struct BenchResult {
    name: String,
    fun: f64,
    nfev: usize,
    nlfev: usize,
    n_local_minima: usize,
    success: bool,
    times_us: Vec<f64>,
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
    fn std_us(&self) -> f64 {
        let mean = self.mean_us();
        let var = self.times_us.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / (self.times_us.len().max(2) - 1) as f64;
        var.sqrt()
    }
}

struct TestConfig {
    name: String,
    sampling_method: SamplingMethod,
    n: usize,
    iters: Option<usize>,
    maxiter: Option<usize>,
    n_runs: usize,
    bounds: Vec<(f64, f64)>,
}

fn bench_with_workers(
    config: &TestConfig,
    func: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    workers: Option<usize>,
) -> BenchResult {
    let mut times_us = Vec::with_capacity(config.n_runs);
    let mut last_fun = 0.0;
    let mut last_nfev = 0;
    let mut last_nlfev = 0;
    let mut last_minima = 0;
    let mut last_success = false;

    for _ in 0..config.n_runs {
        let f = func.clone();
        let b = config.bounds.clone();
        let mut options = ShgoOptions {
            sampling_method: config.sampling_method,
            iters: config.iters,
            maxiter: config.maxiter,
            disp: 0,
            workers,
            ..Default::default()
        };
        if config.n > 0 {
            options.n = config.n;
        }

        let start = Instant::now();
        let wrapper = move |x: &[f64]| f(x);
        let result = Shgo::new(wrapper, b).with_options(options).minimize().unwrap();
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        last_fun = result.fun;
        last_nfev = result.nfev;
        last_nlfev = result.nlfev;
        last_minima = result.xl.len();
        last_success = result.success;
    }

    BenchResult {
        name: config.name.clone(),
        fun: last_fun,
        nfev: last_nfev,
        nlfev: last_nlfev,
        n_local_minima: last_minima,
        success: last_success,
        times_us,
    }
}

fn print_results(label: &str, results: &[BenchResult]) {
    println!("\n{:─<90}", "");
    println!(
        "  {:<32} {:>12} {:>12} {:>12} {:>10} {:>8}",
        label, "median(ms)", "mean(ms)", "min(ms)", "std(ms)", "f(x)"
    );
    println!("{:─<90}", "");
    for r in results {
        let median_ms = r.median_us() / 1000.0;
        let mean_ms = r.mean_us() / 1000.0;
        let min_ms = r.min_us() / 1000.0;
        let std_ms = r.std_us() / 1000.0;
        println!(
            "  {:<32} {:>12.3} {:>12.3} {:>12.3} {:>10.3} {:>8.2e}",
            r.name, median_ms, mean_ms, min_ms, std_ms, r.fun,
        );
    }
}

fn main() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("======================================================================");
    println!("  SHGO-RS Parallel Performance Benchmark");
    println!("  CPUs: {}", num_cpus);
    println!("======================================================================");

    // Warmup
    let _ = Shgo::new(sphere, vec![(-5.0, 5.0); 2]).minimize().unwrap();

    // Define test configurations — longer-running problems
    let configs: Vec<(TestConfig, Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = vec![
        // --- Cheap objective, many local minimizations ---
        (TestConfig {
            name: "rastrigin_5d_sobol_256".into(),
            sampling_method: SamplingMethod::Sobol, n: 256,
            iters: Some(1), maxiter: None, n_runs: 20,
            bounds: vec![(-5.12, 5.12); 5],
        }, Arc::new(rastrigin)),

        (TestConfig {
            name: "rosenbrock_5d_sobol_256".into(),
            sampling_method: SamplingMethod::Sobol, n: 256,
            iters: Some(1), maxiter: None, n_runs: 20,
            bounds: vec![(-5.0, 5.0); 5],
        }, Arc::new(rosenbrock)),

        (TestConfig {
            name: "ackley_5d_sobol_256".into(),
            sampling_method: SamplingMethod::Sobol, n: 256,
            iters: Some(1), maxiter: None, n_runs: 20,
            bounds: vec![(-5.0, 5.0); 5],
        }, Arc::new(ackley)),

        (TestConfig {
            name: "levy_5d_sobol_256".into(),
            sampling_method: SamplingMethod::Sobol, n: 256,
            iters: Some(1), maxiter: None, n_runs: 20,
            bounds: vec![(-10.0, 10.0); 5],
        }, Arc::new(levy)),

        // --- Higher sample counts in 5D ---
        (TestConfig {
            name: "levy_5d_sobol_512".into(),
            sampling_method: SamplingMethod::Sobol, n: 512,
            iters: Some(1), maxiter: None, n_runs: 10,
            bounds: vec![(-10.0, 10.0); 5],
        }, Arc::new(levy)),

        (TestConfig {
            name: "eggholder_2d_sobol_512".into(),
            sampling_method: SamplingMethod::Sobol, n: 512,
            iters: Some(1), maxiter: None, n_runs: 10,
            bounds: vec![(-512.0, 512.0); 2],
        }, Arc::new(eggholder)),

        // --- Expensive objective (~100μs per eval), 2D with Sobol ---
        (TestConfig {
            name: "rastrigin_2d_expensive".into(),
            sampling_method: SamplingMethod::Sobol, n: 128,
            iters: Some(1), maxiter: None, n_runs: 10,
            bounds: vec![(-5.12, 5.12); 2],
        }, Arc::new(make_expensive(rastrigin, 100))),

        // --- Expensive objective (~100μs per eval), 5D with Sobol ---
        (TestConfig {
            name: "rastrigin_5d_expensive".into(),
            sampling_method: SamplingMethod::Sobol, n: 256,
            iters: Some(1), maxiter: None, n_runs: 5,
            bounds: vec![(-5.12, 5.12); 5],
        }, Arc::new(make_expensive(rastrigin, 100))),

        // --- Expensive objective (~500μs per eval), 3D with Sobol ---
        (TestConfig {
            name: "ackley_3d_very_expensive".into(),
            sampling_method: SamplingMethod::Sobol, n: 128,
            iters: Some(1), maxiter: None, n_runs: 5,
            bounds: vec![(-5.0, 5.0); 3],
        }, Arc::new(make_expensive(ackley, 500))),

        // --- Eggholder with more sampling ---
        (TestConfig {
            name: "eggholder_2d_sobol_512".into(),
            sampling_method: SamplingMethod::Sobol, n: 512,
            iters: Some(1), maxiter: None, n_runs: 20,
            bounds: vec![(-512.0, 512.0); 2],
        }, Arc::new(eggholder)),

        // --- Multi-iteration ---
        (TestConfig {
            name: "rastrigin_3d_simp_5iter".into(),
            sampling_method: SamplingMethod::Simplicial, n: 0,
            iters: Some(5), maxiter: None, n_runs: 10,
            bounds: vec![(-5.12, 5.12); 3],
        }, Arc::new(rastrigin)),
    ];

    let mut serial_results: Vec<BenchResult> = Vec::new();
    let mut parallel_results: Vec<BenchResult> = Vec::new();

    for (config, func) in &configs {
        eprint!("  Benchmarking {}... ", config.name);

        // Single-threaded (workers=1)
        let r1 = bench_with_workers(config, func.clone(), Some(1));
        eprint!("serial={:.1}ms ", r1.median_us() / 1000.0);

        // All CPUs (workers=None → rayon global pool)
        let r2 = bench_with_workers(config, func.clone(), None);
        let speedup = r1.median_us() / r2.median_us();
        eprintln!("parallel={:.1}ms  speedup={:.2}x", r2.median_us() / 1000.0, speedup);

        serial_results.push(r1);
        parallel_results.push(r2);
    }

    // ===== Print tables =====
    print_results("Serial (workers=1)", &serial_results);
    print_results(&format!("Parallel (workers=all, {} CPUs)", num_cpus), &parallel_results);

    // ===== Comparison =====
    println!("\n{:─<90}", "");
    println!(
        "  {:<32} {:>12} {:>12} {:>12}",
        "Test", "Serial(ms)", "Parallel(ms)", "Speedup"
    );
    println!("{:─<90}", "");
    let mut log_speedups = Vec::new();
    for (s, p) in serial_results.iter().zip(parallel_results.iter()) {
        let sm = s.median_us() / 1000.0;
        let pm = p.median_us() / 1000.0;
        let speedup = sm / pm;
        log_speedups.push(speedup.ln());
        println!(
            "  {:<32} {:>12.3} {:>12.3} {:>12.2}x",
            s.name, sm, pm, speedup
        );
    }
    let geo_mean = (log_speedups.iter().sum::<f64>() / log_speedups.len() as f64).exp();
    println!("\n  Geometric mean parallel speedup: {:.2}x", geo_mean);

    // ===== Write JSON =====
    let write_json = |results: &[BenchResult], path: &str| {
        let mut json = String::from("[\n");
        for (i, r) in results.iter().enumerate() {
            json.push_str(&format!(
                "  {{\n    \"name\": \"{}\",\n    \"fun\": {:.16e},\n    \"nfev\": {},\n    \"nlfev\": {},\n    \"n_local_minima\": {},\n    \"success\": {},\n    \"n_runs\": {},\n    \"median_us\": {:.2},\n    \"mean_us\": {:.2},\n    \"min_us\": {:.2},\n    \"std_us\": {:.2}\n  }}",
                r.name, r.fun, r.nfev, r.nlfev, r.n_local_minima, r.success,
                r.times_us.len(),
                r.median_us(), r.mean_us(), r.min_us(), r.std_us(),
            ));
            if i < results.len() - 1 { json.push_str(",\n"); }
            json.push('\n');
        }
        json.push_str("]\n");
        std::fs::write(path, &json).expect("Failed to write JSON");
    };

    write_json(&serial_results, "parallel_bench_serial.json");
    write_json(&parallel_results, "parallel_bench_parallel.json");
    println!("\n  Results written to parallel_bench_serial.json and parallel_bench_parallel.json");
}
