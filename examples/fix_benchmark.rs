//! Before/after benchmark for validating the critical fixes in
//! shgo_fable_recommendations.md §3 (O(1) index lookups, nfev accounting, …).
//!
//! Emits one JSON line per case with timing (min over reps) plus the full
//! accuracy fields (fun, x, funl head, nfev, nlfev, nit) so pre-fix and
//! post-fix runs can be diffed exactly.
//!
//! Run: cargo run --release --example fix_benchmark > baseline.jsonl

use shgo::{ConnectivityMethod, OptimizeResult, SamplingMethod, Shgo, ShgoOptions};
use std::time::Instant;

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

fn options(method: SamplingMethod, conn: ConnectivityMethod, n: usize, maxiter: usize) -> ShgoOptions {
    ShgoOptions {
        sampling_method: method,
        connectivity_method: conn,
        n,
        maxiter: Some(maxiter),
        disp: 0,
        ..Default::default()
    }
}

fn report(name: &str, secs: f64, r: &OptimizeResult) {
    let funl_head: Vec<f64> = r.funl.iter().take(8).cloned().collect();
    println!(
        "{}",
        serde_json::json!({
            "name": name,
            "secs": (secs * 1000.0).round() / 1000.0,
            "fun": r.fun,
            "x": r.x,
            "funl_head": funl_head,
            "n_lmin": r.xl.len(),
            "nfev": r.nfev,
            "nlfev": r.nlfev,
            "nit": r.nit,
            "success": r.success,
        })
    );
}

fn run<F>(name: &str, reps: usize, f: F)
where
    F: Fn() -> OptimizeResult,
{
    let mut best = f64::INFINITY;
    let mut result = None;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        if result.is_none() {
            result = Some(r);
        }
    }
    report(name, best, &result.unwrap());
}

fn main() {
    let reps = 2;

    // 1. Quadratic-scan stressor: big V, trivial minimizer pool.
    run("sobol_knn_sphere_6d_16384x2", reps, || {
        Shgo::new(sphere, vec![(-5.0, 5.0); 6])
            .with_options(options(
                SamplingMethod::Sobol,
                ConnectivityMethod::KNearestNeighbors,
                16384,
                2,
            ))
            .minimize()
            .unwrap()
    });

    // 2. Realistic multimodal accuracy case (funl must match pre/post).
    run("sobol_knn_rastrigin_6d_4096x2", reps, || {
        Shgo::new(rastrigin, vec![(-5.12, 5.12); 6])
            .with_options(options(
                SamplingMethod::Sobol,
                ConnectivityMethod::KNearestNeighbors,
                4096,
                2,
            ))
            .minimize()
            .unwrap()
    });

    // 3. Higher dimension, default k = 2*dim+1.
    run("sobol_knn_rastrigin_10d_2048x2", reps, || {
        Shgo::new(rastrigin, vec![(-5.12, 5.12); 10])
            .with_options(options(
                SamplingMethod::Sobol,
                ConnectivityMethod::KNearestNeighbors,
                2048,
                2,
            ))
            .minimize()
            .unwrap()
    });

    // 4. Delaunay path.
    run("sobol_delaunay_rastrigin_4d_1024x2", reps, || {
        Shgo::new(rastrigin, vec![(-5.12, 5.12); 4])
            .with_options(options(
                SamplingMethod::Sobol,
                ConnectivityMethod::Delaunay,
                1024,
                2,
            ))
            .minimize()
            .unwrap()
    });

    // 5. Simplicial path (default n/iters): initial 2^9 corners + one refine_all,
    //    exercises construct_lcb_simplicial and the pending-pool scans.
    run("simplicial_sphere_9d", reps, || {
        Shgo::new(sphere, vec![(-5.0, 5.0); 9])
            .with_options(ShgoOptions {
                disp: 0,
                ..Default::default()
            })
            .minimize()
            .unwrap()
    });

    // 6. Constrained path (feasibility filtering + COBYLA).
    run("constrained_sphere_4d_1024x2", reps, || {
        let constraint = |x: &[f64]| x[0] + x[1] - 1.0; // x0 + x1 >= 1
        Shgo::with_constraints(sphere, vec![(-5.0, 5.0); 4], vec![constraint])
            .with_options(options(
                SamplingMethod::Sobol,
                ConnectivityMethod::KNearestNeighbors,
                1024,
                2,
            ))
            .minimize()
            .unwrap()
    });
}
