//! Comprehensive memory & performance benchmark harness for SHGO-RS.
//!
//! Runs a matrix of optimization problems and reports both timing and memory
//! statistics in a human-readable table and JSON output.
//!
//! Build:
//!   cargo build --example memory_perf_harness --release --features track-alloc
//!
//! Run:
//!   ./target/release/examples/memory_perf_harness
//!
//! Or run a subset:
//!   ./target/release/examples/memory_perf_harness --filter "rastrigin"
//!   ./target/release/examples/memory_perf_harness --quick   # Only small problems

#[cfg(feature = "track-alloc")]
#[global_allocator]
static ALLOC: shgo::alloc_tracker::TrackingAllocator = shgo::alloc_tracker::TrackingAllocator::new();

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

// ─── Test Functions ──────────────────────────────────────────────────────────

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
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
        + 20.0
        + std::f64::consts::E
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

fn eggholder(x: &[f64]) -> f64 {
    let x1 = x[0];
    let x2 = x[1];
    -(x2 + 47.0) * (x1 / 2.0 + x2 + 47.0).abs().sqrt().sin()
        - x1 * (x1 - (x2 + 47.0)).abs().sqrt().sin()
}

fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.9829 * n - x.iter().map(|&xi| xi * (xi.abs().sqrt()).sin()).sum::<f64>()
}

fn levy(x: &[f64]) -> f64 {
    let w: Vec<f64> = x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();
    let n = w.len();
    let term1 = (std::f64::consts::PI * w[0]).sin().powi(2);
    let sum: f64 = w[..n - 1]
        .iter()
        .enumerate()
        .map(|(i, &wi)| {
            (wi - 1.0).powi(2)
                * (1.0 + 10.0 * (std::f64::consts::PI * w[i + 1]).sin().powi(2))
        })
        .sum();
    let term3 = (w[n - 1] - 1.0).powi(2)
        * (1.0 + (2.0 * std::f64::consts::PI * w[n - 1]).sin().powi(2));
    term1 + sum + term3
}

fn styblinski_tang(x: &[f64]) -> f64 {
    x.iter()
        .map(|&xi| xi.powi(4) - 16.0 * xi.powi(2) + 5.0 * xi)
        .sum::<f64>()
        / 2.0
}

// ─── Benchmark Configuration ────────────────────────────────────────────────

#[derive(Clone)]
struct BenchConfig {
    name: &'static str,
    func_name: &'static str,
    dim: usize,
    sampling: SamplingMethod,
    n_samples: usize,
    iters: Option<usize>,
    maxiter: Option<usize>,
    bounds_range: (f64, f64),
    n_runs: usize,
    category: &'static str, // "quick", "medium", "stress"
}

struct BenchResult {
    config: BenchConfig,
    fun: f64,
    nfev: usize,
    nlfev: usize,
    n_local_minima: usize,
    success: bool,
    times_ms: Vec<f64>,
    #[cfg(feature = "track-alloc")]
    peak_bytes: usize,
    #[cfg(feature = "track-alloc")]
    total_allocated: usize,
    #[cfg(feature = "track-alloc")]
    alloc_count: usize,
}

impl BenchResult {
    fn median_ms(&self) -> f64 {
        let mut sorted = self.times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }
    fn min_ms(&self) -> f64 {
        self.times_ms.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn std_ms(&self) -> f64 {
        let mean = self.times_ms.iter().sum::<f64>() / self.times_ms.len() as f64;
        let var = self.times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / (self.times_ms.len() - 1).max(1) as f64;
        var.sqrt()
    }
}

fn get_func(name: &str) -> Box<dyn Fn(&[f64]) -> f64 + Send + Sync> {
    match name {
        "sphere" => Box::new(sphere),
        "rosenbrock" => Box::new(rosenbrock),
        "ackley" => Box::new(ackley),
        "rastrigin" => Box::new(rastrigin),
        "eggholder" => Box::new(eggholder),
        "schwefel" => Box::new(schwefel),
        "levy" => Box::new(levy),
        "styblinski_tang" => Box::new(styblinski_tang),
        _ => panic!("Unknown function: {}", name),
    }
}

fn build_configs() -> Vec<BenchConfig> {
    let mut configs = Vec::new();

    // ── Quick tests (< 100ms each) ──
    for &func in &["sphere", "rosenbrock", "ackley", "rastrigin", "levy", "styblinski_tang"] {
        let br = match func {
            "rastrigin" => (-5.12, 5.12),
            _ => (-5.0, 5.0),
        };
        configs.push(BenchConfig {
            name: func,
            func_name: func,
            dim: 2,
            sampling: SamplingMethod::Simplicial,
            n_samples: 0,
            iters: Some(1),
            maxiter: None,
            bounds_range: br,
            n_runs: 10,
            category: "quick",
        });
    }

    // Eggholder 2D (wider bounds)
    configs.push(BenchConfig {
        name: "eggholder",
        func_name: "eggholder",
        dim: 2,
        sampling: SamplingMethod::Simplicial,
        n_samples: 0,
        iters: Some(3),
        maxiter: None,
        bounds_range: (-512.0, 512.0),
        n_runs: 10,
        category: "quick",
    });

    // 2D Sobol variants
    for &func in &["sphere", "rosenbrock", "rastrigin"] {
        let br = match func {
            "rastrigin" => (-5.12, 5.12),
            _ => (-5.0, 5.0),
        };
        configs.push(BenchConfig {
            name: func,
            func_name: func,
            dim: 2,
            sampling: SamplingMethod::Sobol,
            n_samples: 128,
            iters: Some(1),
            maxiter: None,
            bounds_range: br,
            n_runs: 10,
            category: "quick",
        });
    }

    // ── Medium tests (100ms - 2s each) ──
    // 3D-5D problems
    for &dim in &[3, 5] {
        for &func in &["rosenbrock", "ackley", "rastrigin"] {
            let br = match func {
                "rastrigin" => (-5.12, 5.12),
                _ => (-5.0, 5.0),
            };
            configs.push(BenchConfig {
                name: func,
                func_name: func,
                dim,
                sampling: SamplingMethod::Simplicial,
                n_samples: 0,
                iters: Some(1),
                maxiter: None,
                bounds_range: br,
                n_runs: 5,
                category: "medium",
            });
            configs.push(BenchConfig {
                name: func,
                func_name: func,
                dim,
                sampling: SamplingMethod::Sobol,
                n_samples: 256,
                iters: Some(1),
                maxiter: None,
                bounds_range: br,
                n_runs: 5,
                category: "medium",
            });
        }
    }

    // High-iteration 2D (deep mesh refinement)
    for &iters in &[3, 5] {
        configs.push(BenchConfig {
            name: "rosenbrock",
            func_name: "rosenbrock",
            dim: 2,
            sampling: SamplingMethod::Simplicial,
            n_samples: 0,
            iters: Some(iters),
            maxiter: None,
            bounds_range: (-5.0, 5.0),
            n_runs: 5,
            category: "medium",
        });
    }

    // ── Stress tests (2s - 60s each) ──
    // High-dimensional (kept to 7D max to avoid qhull segfaults at ≥10D)
    for &dim in &[7] {
        for &func in &["rastrigin", "ackley"] {
            let br = match func {
                "rastrigin" => (-5.12, 5.12),
                _ => (-5.0, 5.0),
            };
            configs.push(BenchConfig {
                name: func,
                func_name: func,
                dim,
                sampling: SamplingMethod::Sobol,
                n_samples: 512,
                iters: Some(1),
                maxiter: None,
                bounds_range: br,
                n_runs: 3,
                category: "stress",
            });
        }
    }

    // Large Sobol sampling
    for &n in &[1024, 4096] {
        configs.push(BenchConfig {
            name: "rastrigin",
            func_name: "rastrigin",
            dim: 5,
            sampling: SamplingMethod::Sobol,
            n_samples: n,
            iters: Some(1),
            maxiter: None,
            bounds_range: (-5.12, 5.12),
            n_runs: 3,
            category: "stress",
        });
    }

    // Deep refinement 3D
    configs.push(BenchConfig {
        name: "rosenbrock",
        func_name: "rosenbrock",
        dim: 3,
        sampling: SamplingMethod::Simplicial,
        n_samples: 0,
        iters: Some(7),
        maxiter: None,
        bounds_range: (-5.0, 5.0),
        n_runs: 3,
        category: "stress",
    });

    // Schwefel (deceptive landscape, needs many samples)
    for &dim in &[5, 7] {
        configs.push(BenchConfig {
            name: "schwefel",
            func_name: "schwefel",
            dim,
            sampling: SamplingMethod::Sobol,
            n_samples: 1024,
            iters: Some(1),
            maxiter: None,
            bounds_range: (-500.0, 500.0),
            n_runs: 3,
            category: "stress",
        });
    }

    configs
}

fn run_bench(config: &BenchConfig) -> BenchResult {
    let mut times_ms = Vec::with_capacity(config.n_runs);
    let mut last_fun = 0.0;
    let mut last_nfev = 0;
    let mut last_nlfev = 0;
    let mut last_minima = 0;
    let mut last_success = false;

    #[cfg(feature = "track-alloc")]
    let mut max_peak: usize = 0;
    #[cfg(feature = "track-alloc")]
    let mut max_total_alloc: usize = 0;
    #[cfg(feature = "track-alloc")]
    let mut max_alloc_count: usize = 0;

    for _ in 0..config.n_runs {
        let func = get_func(config.func_name);
        let bounds = vec![config.bounds_range; config.dim];
        let mut options = ShgoOptions {
            sampling_method: config.sampling,
            iters: config.iters,
            maxiter: config.maxiter,
            disp: 0,
            ..Default::default()
        };
        if config.n_samples > 0 {
            options.n = config.n_samples;
        }

        #[cfg(feature = "track-alloc")]
        shgo::alloc_tracker::reset_peak();

        let start = Instant::now();
        let result = Shgo::new(func, bounds).with_options(options).minimize();
        let elapsed = start.elapsed();

        #[cfg(feature = "track-alloc")]
        {
            let snap = shgo::alloc_tracker::snapshot();
            max_peak = max_peak.max(snap.peak_bytes);
            max_total_alloc = max_total_alloc.max(snap.total_allocated);
            max_alloc_count = max_alloc_count.max(snap.alloc_count);
        }

        times_ms.push(elapsed.as_secs_f64() * 1000.0);
        if let Ok(res) = result {
            last_fun = res.fun;
            last_nfev = res.nfev;
            last_nlfev = res.nlfev;
            last_minima = res.xl.len();
            last_success = res.success;
        }
    }

    BenchResult {
        config: config.clone(),
        fun: last_fun,
        nfev: last_nfev,
        nlfev: last_nlfev,
        n_local_minima: last_minima,
        success: last_success,
        times_ms,
        #[cfg(feature = "track-alloc")]
        peak_bytes: max_peak,
        #[cfg(feature = "track-alloc")]
        total_allocated: max_total_alloc,
        #[cfg(feature = "track-alloc")]
        alloc_count: max_alloc_count,
    }
}

fn sampling_str(s: SamplingMethod) -> &'static str {
    match s {
        SamplingMethod::Simplicial => "simp",
        SamplingMethod::Sobol => "sobol",
    }
}

#[cfg(feature = "track-alloc")]
fn fmt_bytes(bytes: usize) -> String {
    shgo::alloc_tracker::AllocSnapshot::fmt_bytes(bytes)
}

fn print_results(results: &[BenchResult]) {
    #[cfg(feature = "track-alloc")]
    let header = format!(
        "  {:<32} {:>6} {:>8} {:>8} {:>8} {:>7} {:>6} {:>10} {:>10} {:>10}",
        "Test", "dim", "med(ms)", "min(ms)", "std(ms)", "nfev", "nmin",
        "peak_mem", "total_alloc", "allocs"
    );
    #[cfg(not(feature = "track-alloc"))]
    let header = format!(
        "  {:<32} {:>6} {:>8} {:>8} {:>8} {:>7} {:>6} {:>8}",
        "Test", "dim", "med(ms)", "min(ms)", "std(ms)", "nfev", "nmin", "f(x)"
    );

    let sep_len = header.len();

    println!("\n{:─>width$}", "", width = sep_len);
    println!("{}", header);
    println!("{:─>width$}", "", width = sep_len);

    let mut last_category = "";
    for r in results {
        if r.config.category != last_category {
            if !last_category.is_empty() {
                println!();
            }
            println!("  ┌─ {} ─┐", r.config.category.to_uppercase());
            last_category = r.config.category;
        }

        let label = format!(
            "{}_{}d_{}_n{}{}",
            r.config.name,
            r.config.dim,
            sampling_str(r.config.sampling),
            if r.config.n_samples > 0 {
                r.config.n_samples.to_string()
            } else {
                "auto".to_string()
            },
            if r.config.iters.unwrap_or(1) > 1 {
                format!("_i{}", r.config.iters.unwrap())
            } else {
                String::new()
            }
        );

        #[cfg(feature = "track-alloc")]
        println!(
            "  {:<32} {:>6} {:>8.2} {:>8.2} {:>8.2} {:>7} {:>6} {:>10} {:>10} {:>10}",
            label,
            r.config.dim,
            r.median_ms(),
            r.min_ms(),
            r.std_ms(),
            r.nfev,
            r.n_local_minima,
            fmt_bytes(r.peak_bytes),
            fmt_bytes(r.total_allocated),
            r.alloc_count,
        );

        #[cfg(not(feature = "track-alloc"))]
        println!(
            "  {:<32} {:>6} {:>8.2} {:>8.2} {:>8.2} {:>7} {:>6} {:>8.2e}",
            label,
            r.config.dim,
            r.median_ms(),
            r.min_ms(),
            r.std_ms(),
            r.nfev,
            r.n_local_minima,
            r.fun,
        );
    }
    println!();
}

fn write_json(results: &[BenchResult], path: &str) {
    let mut json = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        let label = format!(
            "{}_{}d_{}_n{}",
            r.config.name,
            r.config.dim,
            sampling_str(r.config.sampling),
            if r.config.n_samples > 0 {
                r.config.n_samples.to_string()
            } else {
                "auto".to_string()
            }
        );

        json.push_str(&format!(
            "  {{\n    \"label\": \"{}\",\n    \"function\": \"{}\",\n    \"dim\": {},\n    \"sampling\": \"{}\",\n    \"n_samples\": {},\n    \"iters\": {},\n    \"category\": \"{}\",\n    \"fun\": {:.16e},\n    \"nfev\": {},\n    \"nlfev\": {},\n    \"n_local_minima\": {},\n    \"success\": {},\n    \"n_runs\": {},\n    \"median_ms\": {:.4},\n    \"min_ms\": {:.4},\n    \"std_ms\": {:.4}",
            label,
            r.config.func_name,
            r.config.dim,
            sampling_str(r.config.sampling),
            r.config.n_samples,
            r.config.iters.unwrap_or(1),
            r.config.category,
            r.fun,
            r.nfev,
            r.nlfev,
            r.n_local_minima,
            r.success,
            r.config.n_runs,
            r.median_ms(),
            r.min_ms(),
            r.std_ms(),
        ));

        #[cfg(feature = "track-alloc")]
        json.push_str(&format!(
            ",\n    \"peak_bytes\": {},\n    \"total_allocated\": {},\n    \"alloc_count\": {}",
            r.peak_bytes, r.total_allocated, r.alloc_count,
        ));

        json.push_str(&format!(
            ",\n    \"all_times_ms\": [{}]\n  }}",
            r.times_ms
                .iter()
                .map(|t| format!("{:.4}", t))
                .collect::<Vec<_>>()
                .join(", "),
        ));

        if i < results.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("]\n");

    std::fs::write(path, &json).expect("Failed to write JSON");
    println!("  Results written to {}", path);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filter = args.iter().find(|a| !a.starts_with("--")).and_then(|_| None::<&str>)
        .or_else(|| {
            args.iter()
                .position(|a| a == "--filter")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
        });
    let quick_only = args.iter().any(|a| a == "--quick");
    let no_stress = args.iter().any(|a| a == "--no-stress");

    let configs = build_configs();
    let configs: Vec<_> = configs
        .into_iter()
        .filter(|c| {
            if quick_only && c.category != "quick" {
                return false;
            }
            if no_stress && c.category == "stress" {
                return false;
            }
            if let Some(f) = filter {
                return c.name.contains(f) || c.func_name.contains(f);
            }
            true
        })
        .collect();

    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("══════════════════════════════════════════════════════════════════════");
    println!("  SHGO-RS Memory & Performance Harness");
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  CPU cores: {}", num_cpus);
    println!("  Tests: {}", configs.len());
    #[cfg(feature = "track-alloc")]
    println!("  Memory tracking: ENABLED");
    #[cfg(not(feature = "track-alloc"))]
    println!("  Memory tracking: DISABLED (enable with --features track-alloc)");
    println!();

    // Warmup
    let _ = Shgo::new(sphere, vec![(-5.0, 5.0); 2])
        .minimize();

    let mut results = Vec::new();
    for (i, config) in configs.iter().enumerate() {
        let label = format!(
            "{}_{}d_{}_n{}",
            config.name,
            config.dim,
            sampling_str(config.sampling),
            if config.n_samples > 0 {
                config.n_samples.to_string()
            } else {
                "auto".to_string()
            }
        );
        eprint!(
            "  [{}/{}] Running {} ({} runs)...",
            i + 1,
            configs.len(),
            label,
            config.n_runs,
        );

        let result = run_bench(config);
        eprintln!(" {:.2} ms (median)", result.median_ms());
        results.push(result);
    }

    print_results(&results);
    write_json(&results, "memory_perf_results.json");
}
