//! Difficult high-dimensional benchmark for SHGO
//!
//! Tests SHGO's ability to find global minima in challenging landscapes
//! without relying on local optimization.
//!
//! Run with: cargo run --example difficult_benchmark --release

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Schwefel function - notoriously difficult for global optimization
///
/// Properties:
/// - Global minimum at x_i = 420.9687... for all i
/// - f(x*) ≈ 0 (we shift it so minimum is 0)
/// - Many deceptive local minima
/// - The second-best minimum is FAR from the global minimum
/// - Search space: [-500, 500]^n
///
/// This function is designed to fool optimization algorithms because
/// the global minimum is near the boundary and local search from most
/// starting points will find inferior local minima.
fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|&xi| xi * (xi.abs().sqrt()).sin()).sum();
    418.9829 * n - sum
}

/// Expensive Schwefel with computational work to simulate real-world cost
fn expensive_schwefel(x: &[f64]) -> f64 {
    // Add computational work
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    
    schwefel(x) + extra_work * 1e-20
}

/// Noisy Schwefel - adds 2% multiplicative noise
/// This tests robustness to measurement/simulation noise
fn noisy_schwefel(x: &[f64]) -> f64 {
    // Use atomic counter for deterministic but varying "random" noise
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    
    // Simple pseudo-random based on counter and coordinates
    let mut hash = count;
    for &xi in x.iter() {
        hash ^= (xi.to_bits() as u64).wrapping_mul(0x9e3779b97f4a7c15);
        hash = hash.wrapping_mul(0x517cc1b727220a95);
    }
    
    // Convert to [-1, 1] range
    let noise_factor = ((hash as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
    
    // Add computational work  
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    
    let base_value = schwefel(x);
    
    // 2% multiplicative noise (relative to function value magnitude)
    // For values near 0, use additive noise based on typical scale
    let noise_scale = if base_value.abs() > 100.0 {
        base_value.abs() * 0.02  // 2% of value
    } else {
        2.0  // Additive noise when value is small
    };
    
    base_value + noise_factor * noise_scale + extra_work * 1e-20
}

/// Levy function - another challenging multimodal function
/// Global minimum at x_i = 1 for all i, f(x*) = 0
fn levy(x: &[f64]) -> f64 {
    let w: Vec<f64> = x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();
    let n = w.len();
    
    let term1 = (std::f64::consts::PI * w[0]).sin().powi(2);
    
    let sum: f64 = w[..n-1].iter().enumerate().map(|(i, &wi)| {
        (wi - 1.0).powi(2) * (1.0 + 10.0 * (std::f64::consts::PI * w[i+1]).sin().powi(2))
    }).sum();
    
    let term3 = (w[n-1] - 1.0).powi(2) * (1.0 + (2.0 * std::f64::consts::PI * w[n-1]).sin().powi(2));
    
    term1 + sum + term3
}

/// Expensive Levy
fn expensive_levy(x: &[f64]) -> f64 {
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    levy(x) + extra_work * 1e-20
}

fn run_benchmark<F>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    n_samples: usize,
    expected_min: f64,
    expected_x: Option<Vec<f64>>,
) where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    let dim = bounds.len();
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("\n=== {} ({}D) ===", name, dim);
    println!("Expected minimum: f ≈ {:.4}", expected_min);
    if let Some(ref x) = expected_x {
        if x.len() <= 5 {
            println!("Expected location: x ≈ {:?}", x);
        } else {
            println!("Expected location: x_i ≈ {:.4} for all i", x[0]);
        }
    }
    println!();

    let options = ShgoOptions {
        n: n_samples,
        maxiter: Some(3),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: false,  // No local optimization!
        workers: None,  // Use all cores
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(func, bounds)
        .with_options(options)
        .minimize();
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            let error = (res.fun - expected_min).abs();
            let success = error < expected_min.abs() * 0.1 + 1.0; // Within 10% or 1.0 absolute
            
            println!("  Result: f = {:.6e}", res.fun);
            println!("  Error from optimum: {:.6e}", error);
            if dim <= 5 {
                println!("  Location: x = {:?}", res.x);
            } else {
                println!("  Location: x[0..3] = [{:.4}, {:.4}, {:.4}, ...]", 
                    res.x[0], res.x[1], res.x[2]);
            }
            println!("  Function evaluations: {}", res.nfev);
            println!("  Time: {:.3}s ({} cores)", elapsed.as_secs_f64(), num_cpus);
            println!("  Success: {}", if success { "✓ FOUND GLOBAL MINIMUM" } else { "✗ Found local minimum" });
        }
        Err(e) => {
            println!("  Error: {:?}", e);
        }
    }
}

/// Run benchmark WITH local optimization enabled
fn run_benchmark_with_local<F>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    n_samples: usize,
    expected_min: f64,
    expected_x: Option<Vec<f64>>,
) where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    let dim = bounds.len();
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("\n=== {} ({}D) ===", name, dim);
    println!("Expected minimum: f ≈ {:.4}", expected_min);
    if let Some(ref x) = expected_x {
        if x.len() <= 5 {
            println!("Expected location: x ≈ {:?}", x);
        } else {
            println!("Expected location: x_i ≈ {:.4} for all i", x[0]);
        }
    }
    println!();

    let options = ShgoOptions {
        n: n_samples,
        maxiter: Some(3),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: true,  // WITH local optimization!
        workers: None,
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(func, bounds)
        .with_options(options)
        .minimize();
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            let error = (res.fun - expected_min).abs();
            let success = error < expected_min.abs() * 0.1 + 1.0;
            
            println!("  Result: f = {:.6e}", res.fun);
            println!("  Error from optimum: {:.6e}", error);
            if dim <= 5 {
                println!("  Location: x = {:?}", res.x);
            } else {
                println!("  Location: x[0..3] = [{:.4}, {:.4}, {:.4}, ...]", 
                    res.x[0], res.x[1], res.x[2]);
            }
            println!("  Function evaluations: {}", res.nfev);
            println!("  Local minimizations: {}", res.nlfev);
            println!("  Time: {:.3}s ({} cores)", elapsed.as_secs_f64(), num_cpus);
            println!("  Success: {}", if success { "✓ FOUND GLOBAL MINIMUM" } else { "✗ Found local minimum" });
        }
        Err(e) => {
            println!("  Error: {:?}", e);
        }
    }
}

/// Run benchmark with LIGHT local optimization - limit effort
fn run_benchmark_light_local<F>(
    name: &str,
    func: F,
    bounds: Vec<(f64, f64)>,
    n_samples: usize,
    expected_min: f64,
    expected_x: Option<Vec<f64>>,
) where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    use shgo::LocalOptimizerOptions;
    
    let dim = bounds.len();
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("\n=== {} ({}D) ===", name, dim);
    println!("Expected minimum: f ≈ {:.4}", expected_min);
    if let Some(ref x) = expected_x {
        if x.len() <= 5 {
            println!("Expected location: x ≈ {:?}", x);
        } else {
            println!("Expected location: x_i ≈ {:.4} for all i", x[0]);
        }
    }
    println!();

    // Light local optimization settings
    let local_options = LocalOptimizerOptions {
        maxeval: Some(50),      // Only 50 evals per local opt (vs default 1000)
        ftol_rel: 1e-4,         // Looser tolerance (vs default 1e-8)
        xtol_rel: 1e-4,         // Looser tolerance
        ..Default::default()
    };

    let options = ShgoOptions {
        n: n_samples,
        maxiter: Some(3),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: true,
        maxiter_local: Some(32),  // Only refine top 32 candidates!
        local_options,
        workers: None,
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(func, bounds)
        .with_options(options)
        .minimize();
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            let error = (res.fun - expected_min).abs();
            let success = error < expected_min.abs() * 0.1 + 1.0;
            
            println!("  Result: f = {:.6e}", res.fun);
            println!("  Error from optimum: {:.6e}", error);
            if dim <= 5 {
                println!("  Location: x = {:?}", res.x);
            } else {
                println!("  Location: x[0..3] = [{:.4}, {:.4}, {:.4}, ...]", 
                    res.x[0], res.x[1], res.x[2]);
            }
            println!("  Function evaluations: {}", res.nfev);
            println!("  Local minimizations: {}", res.nlfev);
            println!("  Time: {:.3}s ({} cores)", elapsed.as_secs_f64(), num_cpus);
            println!("  Success: {}", if success { "✓ FOUND GLOBAL MINIMUM" } else { "✗ Found local minimum" });
        }
        Err(e) => {
            println!("  Error: {:?}", e);
        }
    }
}

fn main() {
    println!("=======================================================");
    println!("  DIFFICULT HIGH-DIMENSIONAL BENCHMARK");
    println!("  Testing SHGO without local optimization");
    println!("=======================================================");
    
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    println!("\nUsing {} CPU cores for parallel evaluation", num_cpus);

    // ============================================================
    // Test 1: Schwefel function - increasing dimensions
    // ============================================================
    println!("\n\n### SCHWEFEL FUNCTION ###");
    println!("Known to be very difficult - global minimum is near boundary");
    println!("and far from most local minima.");

    // Schwefel bounds: [-500, 500], optimal at ~420.9687
    for dim in [5, 10, 15, 20] {
        let bounds = vec![(-500.0, 500.0); dim];
        let expected_x = vec![420.9687; dim];
        run_benchmark(
            "Schwefel (clean)",
            expensive_schwefel,
            bounds,
            1024,  // n samples
            0.0,   // expected minimum (we shifted it)
            Some(expected_x),
        );
    }

    // ============================================================
    // Test 2: Schwefel with 2% noise
    // ============================================================
    println!("\n\n### SCHWEFEL FUNCTION WITH 2% NOISE ###");
    println!("Testing robustness to noisy function evaluations");
    
    for dim in [5, 10, 15, 20] {
        let bounds = vec![(-500.0, 500.0); dim];
        let expected_x = vec![420.9687; dim];
        run_benchmark(
            "Schwefel (2% noise)",
            noisy_schwefel,
            bounds,
            1024,
            0.0,
            Some(expected_x),
        );
    }

    // ============================================================
    // Test 3: Levy function for comparison
    // ============================================================
    println!("\n\n### LEVY FUNCTION ###");
    println!("Another challenging multimodal function");
    
    for dim in [5, 10, 15, 20] {
        let bounds = vec![(-10.0, 10.0); dim];
        let expected_x = vec![1.0; dim];
        run_benchmark(
            "Levy (clean)",
            expensive_levy,
            bounds,
            1024,
            0.0,
            Some(expected_x),
        );
    }

    // ============================================================
    // Test 4: WITH local optimization for comparison
    // ============================================================
    println!("\n\n### SCHWEFEL WITH LOCAL OPTIMIZATION ###");
    println!("Showing the importance of local refinement");
    
    for dim in [5, 10] {
        let bounds = vec![(-500.0, 500.0); dim];
        let expected_x = vec![420.9687; dim];
        run_benchmark_with_local(
            "Schwefel + Local Opt (full)",
            expensive_schwefel,
            bounds,
            256,  // fewer samples, but with local opt
            0.0,
            Some(expected_x),
        );
    }

    // ============================================================
    // Test 5: LIGHT local optimization - limit effort
    // ============================================================
    println!("\n\n### SCHWEFEL WITH LIGHT LOCAL OPTIMIZATION ###");
    println!("Limiting local opt effort so it's not the dominant cost");
    println!("Options:");
    println!("  - maxiter_local: 32 (only refine top 32 candidates)");
    println!("  - maxeval: 50 per local opt (vs default 1000)");
    println!("  - Looser tolerances (1e-4 vs 1e-8)");
    
    for dim in [5, 10, 15, 20] {
        let bounds = vec![(-500.0, 500.0); dim];
        let expected_x = vec![420.9687; dim];
        run_benchmark_light_local(
            "Schwefel + Light Local",
            expensive_schwefel,
            bounds,
            512,  // more global samples
            0.0,
            Some(expected_x),
        );
    }

    // ============================================================
    // Summary
    // ============================================================
    println!("\n\n=======================================================");
    println!("  ANALYSIS");
    println!("=======================================================");
    println!();
    println!("Without local optimization, SHGO relies purely on sampling");
    println!("to find good regions. For difficult functions like Schwefel:");
    println!();
    println!("1. The global minimum may not be sampled directly");
    println!("2. Higher dimensions require exponentially more samples");
    println!("3. Noise further complicates finding the true minimum");
    println!();
    println!("This demonstrates why local optimization is important for");
    println!("refining coarse global search results.");
}
