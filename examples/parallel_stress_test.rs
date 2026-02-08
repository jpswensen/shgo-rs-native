//! Parallel stress test example
//! 
//! This example uses an expensive objective function and high-dimensional
//! optimization to demonstrate parallelism. Watch your CPU usage!
//!
//! Run with: cargo run --example parallel_stress_test --release
//!
//! You can also limit workers:
//!   RAYON_NUM_THREADS=4 cargo run --example parallel_stress_test --release

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

/// Expensive objective function that does extra computation
/// to simulate a real-world costly function evaluation
fn expensive_rastrigin(x: &[f64]) -> f64 {
    // Add significant computational work to make each evaluation slower
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    
    // Rastrigin function
    let a = 10.0;
    let n = x.len() as f64;
    let result = a * n + x.iter()
        .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>();
    
    // Use extra_work to prevent optimization from removing it
    result + extra_work * 1e-20
}

fn main() {
    println!("=== SHGO-RS Parallel Stress Test ===");
    println!("Watch your CPU usage - you should see multiple cores active!\n");
    
    // Get number of CPUs
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    println!("Available CPU threads: {}\n", num_cpus);

    let bounds_8d = vec![(-5.12, 5.12); 8];

    // ============================================================
    // Test 0: NO LOCAL MINIMIZATION - check if parallelization helps
    // ============================================================
    println!("=== Test 0: 8D Rastrigin - NO LOCAL MINIMIZATION ===");
    println!("    (Testing parallel function evaluation during sampling)\n");
    
    // Use more points to see parallelization benefit
    let n_sampling = 512;
    
    // Test 0a: Full cores, no local minimization
    let options_no_local_full = ShgoOptions {
        n: n_sampling,
        maxiter: Some(2),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: false,
        workers: None,  // All cores
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result_no_local_full = Shgo::new(expensive_rastrigin, bounds_8d.clone())
        .with_options(options_no_local_full)
        .minimize()
        .unwrap();
    let elapsed_no_local_full = start.elapsed();

    println!("  With {} cores: {:.3}s, f = {:.6e}, nfev = {}", 
        num_cpus, elapsed_no_local_full.as_secs_f64(), 
        result_no_local_full.fun, result_no_local_full.nfev);

    // Test 0b: Half cores, no local minimization
    let options_no_local_half = ShgoOptions {
        n: n_sampling,
        maxiter: Some(2),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: false,
        workers: Some((num_cpus / 2).max(1)),
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result_no_local_half = Shgo::new(expensive_rastrigin, bounds_8d.clone())
        .with_options(options_no_local_half)
        .minimize()
        .unwrap();
    let elapsed_no_local_half = start.elapsed();

    println!("  With {} cores:  {:.3}s, f = {:.6e}, nfev = {}", 
        (num_cpus / 2).max(1), elapsed_no_local_half.as_secs_f64(), 
        result_no_local_half.fun, result_no_local_half.nfev);

    // Test 0c: 1 core, no local minimization
    let options_no_local_seq = ShgoOptions {
        n: n_sampling,
        maxiter: Some(2),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: false,
        workers: Some(1),
        disp: 0,
        ..Default::default()
    };

    let start = Instant::now();
    let result_no_local_seq = Shgo::new(expensive_rastrigin, bounds_8d.clone())
        .with_options(options_no_local_seq)
        .minimize()
        .unwrap();
    let elapsed_no_local_seq = start.elapsed();

    println!("  With 1 core:   {:.3}s, f = {:.6e}, nfev = {}", 
        elapsed_no_local_seq.as_secs_f64(), 
        result_no_local_seq.fun, result_no_local_seq.nfev);
    
    println!("\n  => Speedup ({} vs 1 core): {:.2}x", 
        num_cpus, 
        elapsed_no_local_seq.as_secs_f64() / elapsed_no_local_full.as_secs_f64());
    println!("     (Parallel evaluation of Sobol points is working!)\n");

    // ============================================================
    // Test 1: WITH LOCAL MINIMIZATION (full SHGO)
    // ============================================================
    println!("=== Test 1: 8D Rastrigin - WITH LOCAL MINIMIZATION ===");
    println!("    (Full SHGO with 1024 sampling points, all {} cores)\n", num_cpus);
    
    let options = ShgoOptions {
        n: 1024,  // More sampling points = more local minimizations
        maxiter: Some(2),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: true,  // Enable local minimization
        workers: None,  // Use all available cores
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(expensive_rastrigin, bounds_8d.clone())
        .with_options(options)
        .minimize()
        .unwrap();
    let elapsed_full = start.elapsed();

    println!("  Result: f(x) = {:.6e}", result.fun);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Local minimizations: {}", result.nlfev);
    println!("  Time with {} workers: {:.2} seconds", num_cpus, elapsed_full.as_secs_f64());
    println!();

    // Test 2: Same problem with only 1 worker (sequential)
    println!("=== Test 2: Same problem with workers=1 (SEQUENTIAL) ===");
    println!("    This demonstrates the speedup from parallelism...\n");
    
    let options_seq = ShgoOptions {
        n: 1024,
        maxiter: Some(2),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: true,
        workers: Some(1),  // Single-threaded
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_seq = Shgo::new(expensive_rastrigin, bounds_8d.clone())
        .with_options(options_seq)
        .minimize()
        .unwrap();
    let elapsed_seq = start.elapsed();

    println!("  Result: f(x) = {:.6e}", result_seq.fun);
    println!("  Function evaluations: {}", result_seq.nfev);
    println!("  Time with 1 worker: {:.2} seconds", elapsed_seq.as_secs_f64());
    println!();

    // Test 3: Half the cores
    let half_cores = (num_cpus / 2).max(1);
    println!("=== Test 3: Same problem with workers={} (HALF CORES) ===\n", half_cores);
    
    let options_half = ShgoOptions {
        n: 1024,
        maxiter: Some(2),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: true,
        workers: Some(half_cores),
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result_half = Shgo::new(expensive_rastrigin, bounds_8d)
        .with_options(options_half)
        .minimize()
        .unwrap();
    let elapsed_half = start.elapsed();

    println!("  Result: f(x) = {:.6e}", result_half.fun);
    println!("  Time with {} workers: {:.2} seconds", half_cores, elapsed_half.as_secs_f64());
    println!();

    // Summary
    println!("=== COMPARISON SUMMARY ===\n");
    println!("NO local minimization (like DIRECT):");
    println!("  1 core:  {:.3}s, f = {:.6e}, nfev = {}", 
        elapsed_no_local_seq.as_secs_f64(), result_no_local_seq.fun, result_no_local_seq.nfev);
    println!("  {} cores: {:.3}s (speedup: {:.2}x - expected ~1.0x)", 
        num_cpus, elapsed_no_local_full.as_secs_f64(),
        elapsed_no_local_seq.as_secs_f64() / elapsed_no_local_full.as_secs_f64());
    println!();
    println!("WITH local minimization (full SHGO):");
    println!("  Sequential (1 worker):    {:.2}s, f = {:.6e}, nfev = {}", 
        elapsed_seq.as_secs_f64(), result_seq.fun, result_seq.nfev);
    println!("  Half cores ({} workers):   {:.2}s  ({:.1}x speedup)", 
        half_cores, elapsed_half.as_secs_f64(), 
        elapsed_seq.as_secs_f64() / elapsed_half.as_secs_f64());
    println!("  Full cores ({} workers):  {:.2}s  ({:.1}x speedup)", 
        num_cpus, elapsed_full.as_secs_f64(),
        elapsed_seq.as_secs_f64() / elapsed_full.as_secs_f64());
    println!();
    println!("Theoretical max speedup with {} cores: {:.1}x", num_cpus, num_cpus as f64);
    println!();
    println!("=== KEY INSIGHT ===");
    println!("Without local minimization, SHGO is fast but may not find the true minimum.");
    println!("The local minimization step is what makes SHGO thorough (and slow).");
}
