//! Side-by-side comparison benchmark for SHGO-RS vs Python SHGO.
//!
//! Runs the same test functions with identical parameters and reports
//! results in JSON + human-readable format for comparison.
//!
//! Run with: cargo run --example compare_with_python --release

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
        .map(|(&wi, &wi1)| (wi - 1.0).powi(2) * (1.0 + 10.0 * (std::f64::consts::PI * wi1).sin().powi(2)))
        .sum();
    let term3 = (w[n - 1] - 1.0).powi(2)
        * (1.0 + (2.0 * std::f64::consts::PI * w[n - 1]).sin().powi(2));
    term1 + sum + term3
}

// ===== Result type =====

struct TestResult {
    name: String,
    x: Vec<f64>,
    fun: f64,
    nfev: usize,
    nlfev: usize,
    nit: usize,
    n_local_minima: usize,
    success: bool,
    time_ms: f64,
}

fn print_result(r: &TestResult, label: &str) {
    let x_str = if r.x.len() <= 5 {
        format!(
            "[{}]",
            r.x.iter()
                .map(|xi| format!("{:.8}", xi))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        format!(
            "[{:.8}, {:.8}, ..., {:.8}]",
            r.x[0],
            r.x[1],
            r.x[r.x.len() - 1]
        )
    };

    println!("  {}:", label);
    println!("    x       = {}", x_str);
    println!("    f(x)    = {:.10e}", r.fun);
    println!("    nfev    = {}", r.nfev);
    println!("    nlfev   = {}", r.nlfev);
    println!("    nit     = {}", r.nit);
    println!("    minima  = {}", r.n_local_minima);
    println!("    success = {}", r.success);
    println!("    time    = {:.2} ms", r.time_ms);
}

fn run_test<F: Fn(&[f64]) -> f64 + Send + Sync + 'static>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    sampling_method: SamplingMethod,
    n: usize,
    iters: Option<usize>,
    maxiter: Option<usize>,
) -> TestResult {
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
    let result = Shgo::new(func, bounds)
        .with_options(options)
        .minimize()
        .unwrap();
    let elapsed = start.elapsed();

    TestResult {
        name: name.to_string(),
        x: result.x,
        fun: result.fun,
        nfev: result.nfev,
        nlfev: result.nlfev,
        nit: result.nit,
        n_local_minima: result.xl.len(),
        success: result.success,
        time_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

fn run_test_constrained<
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    G: Fn(&[f64]) -> f64 + Send + Sync + 'static,
>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    constraints: Vec<G>,
    sampling_method: SamplingMethod,
    n: usize,
    iters: Option<usize>,
    maxiter: Option<usize>,
) -> TestResult {
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
    let result = Shgo::with_constraints(func, bounds, constraints)
        .with_options(options)
        .minimize()
        .unwrap();
    let elapsed = start.elapsed();

    TestResult {
        name: name.to_string(),
        x: result.x,
        fun: result.fun,
        nfev: result.nfev,
        nlfev: result.nlfev,
        nit: result.nit,
        n_local_minima: result.xl.len(),
        success: result.success,
        time_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("  RUST SHGO — Comparison Benchmark");
    println!("{}", "=".repeat(70));

    let mut all_results: Vec<TestResult> = Vec::new();

    // ===== Test 1: Sphere 2D, simplicial, default =====
    println!("\n--- Test 1: Sphere 2D (simplicial, default) ---");
    println!("  Expected: x = (0, 0), f(x) = 0");
    let r = run_test(
        "sphere_2d_simp", sphere,
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 2: Sphere 2D, sobol =====
    println!("\n--- Test 2: Sphere 2D (sobol, n=64) ---");
    println!("  Expected: x = (0, 0), f(x) = 0");
    let r = run_test(
        "sphere_2d_sobol", sphere,
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        SamplingMethod::Sobol, 64, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 3: Rosenbrock 2D, simplicial =====
    println!("\n--- Test 3: Rosenbrock 2D (simplicial, default) ---");
    println!("  Expected: x = (1, 1), f(x) = 0");
    let r = run_test(
        "rosenbrock_2d_simp", rosenbrock,
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 4: Rosenbrock 2D, sobol =====
    println!("\n--- Test 4: Rosenbrock 2D (sobol, n=128) ---");
    println!("  Expected: x = (1, 1), f(x) = 0");
    let r = run_test(
        "rosenbrock_2d_sobol", rosenbrock,
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        SamplingMethod::Sobol, 128, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 5: Ackley 2D =====
    println!("\n--- Test 5: Ackley 2D (simplicial, default) ---");
    println!("  Expected: x = (0, 0), f(x) = 0");
    let r = run_test(
        "ackley_2d_simp", ackley,
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 6: Rastrigin 2D, simplicial =====
    println!("\n--- Test 6: Rastrigin 2D (simplicial, default) ---");
    println!("  Expected: x = (0, 0), f(x) = 0");
    let r = run_test(
        "rastrigin_2d_simp", rastrigin,
        vec![(-5.12, 5.12), (-5.12, 5.12)],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 7: Rastrigin 2D, sobol =====
    println!("\n--- Test 7: Rastrigin 2D (sobol, n=128) ---");
    println!("  Expected: x = (0, 0), f(x) = 0");
    let r = run_test(
        "rastrigin_2d_sobol", rastrigin,
        vec![(-5.12, 5.12), (-5.12, 5.12)],
        SamplingMethod::Sobol, 128, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 8: Eggholder 2D, simplicial, iters=3 =====
    println!("\n--- Test 8: Eggholder 2D (simplicial, iters=3) ---");
    println!("  Expected: x ≈ (512, 404.23), f(x) ≈ -959.64");
    let r = run_test(
        "eggholder_2d_simp", eggholder,
        vec![(-512.0, 512.0), (-512.0, 512.0)],
        SamplingMethod::Simplicial, 0, Some(3), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 9: Eggholder 2D, sobol =====
    println!("\n--- Test 9: Eggholder 2D (sobol, n=256) ---");
    println!("  Expected: x ≈ (512, 404.23), f(x) ≈ -959.64");
    let r = run_test(
        "eggholder_2d_sobol", eggholder,
        vec![(-512.0, 512.0), (-512.0, 512.0)],
        SamplingMethod::Sobol, 256, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 10: Styblinski-Tang 2D =====
    println!("\n--- Test 10: Styblinski-Tang 2D (simplicial, default) ---");
    println!("  Expected: x = (-2.9035, -2.9035), f(x) ≈ -78.332");
    let r = run_test(
        "styblinski_2d_simp", styblinski_tang,
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 11: Levy 2D =====
    println!("\n--- Test 11: Levy 2D (simplicial, default) ---");
    println!("  Expected: x = (1, 1), f(x) = 0");
    let r = run_test(
        "levy_2d_simp", levy,
        vec![(-10.0, 10.0), (-10.0, 10.0)],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 12: Sphere 5D, simplicial =====
    println!("\n--- Test 12: Sphere 5D (simplicial, default) ---");
    println!("  Expected: x = (0,...,0), f(x) = 0");
    let r = run_test(
        "sphere_5d_simp", sphere,
        vec![(-5.0, 5.0); 5],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 13: Rosenbrock 3D, simplicial =====
    println!("\n--- Test 13: Rosenbrock 3D (simplicial, default) ---");
    println!("  Expected: x = (1, 1, 1), f(x) = 0");
    let r = run_test(
        "rosenbrock_3d_simp", rosenbrock,
        vec![(-5.0, 5.0); 3],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 14: Constrained sphere =====
    println!("\n--- Test 14: Sphere 2D constrained (x0+x1 >= 1) ---");
    println!("  Expected: x = (0.5, 0.5), f(x) = 0.5");
    let constraint = |x: &[f64]| x[0] + x[1] - 1.0;
    let r = run_test_constrained(
        "sphere_2d_constrained", sphere,
        vec![(0.0, 2.0), (0.0, 2.0)],
        vec![constraint],
        SamplingMethod::Simplicial, 0, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Test 15: Rastrigin 3D, sobol =====
    println!("\n--- Test 15: Rastrigin 3D (sobol, n=128) ---");
    println!("  Expected: x = (0, 0, 0), f(x) = 0");
    let r = run_test(
        "rastrigin_3d_sobol", rastrigin,
        vec![(-5.12, 5.12); 3],
        SamplingMethod::Sobol, 128, Some(1), None,
    );
    print_result(&r, "Rust");
    all_results.push(r);

    // ===== Summary =====
    println!("\n{}", "=".repeat(70));
    println!("  SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        "  {:<30} {:<16} {:<8} {:<8} {:<8} {}",
        "Test", "f(x)", "nfev", "nlfev", "minima", "ok"
    );
    println!("  {}", "-".repeat(78));
    for r in &all_results {
        let ok = if r.success { "✓" } else { "✗" };
        println!(
            "  {:<30} {:<16.6e} {:<8} {:<8} {:<8} {}",
            r.name, r.fun, r.nfev, r.nlfev, r.n_local_minima, ok
        );
    }

    // Write JSON for comparison
    println!("\n  Writing rust_results.json...");
    let json_results: Vec<serde_json::Value> = all_results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "x": r.x,
                "fun": r.fun,
                "nfev": r.nfev,
                "nlfev": r.nlfev,
                "nit": r.nit,
                "n_local_minima": r.n_local_minima,
                "success": r.success,
                "time_ms": r.time_ms,
            })
        })
        .collect();
    let json_str = serde_json::to_string_pretty(&json_results).unwrap();
    std::fs::write("rust_results.json", json_str).unwrap();
    println!("  Done.");
}
