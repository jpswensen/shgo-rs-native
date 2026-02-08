//! Quick test: Schwefel 10D with high sampling, no local opt

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

fn expensive_schwefel(x: &[f64]) -> f64 {
    // Add computational work
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|&xi| xi * (xi.abs().sqrt()).sin()).sum();
    418.9829 * n - sum + extra_work * 1e-20
}

fn main() {
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    
    println!("=== Schwefel 10D - High Sampling, No Local Opt ===");
    println!("n = 2000, maxiter = 10");
    println!("Using {} CPU cores\n", num_cpus);
    
    let bounds = vec![(-500.0, 500.0); 10];
    
    let options = ShgoOptions {
        n: 2000,
        maxiter: Some(10),
        sampling_method: SamplingMethod::Sobol,
        minimize_every_iter: false,  // No local optimization!
        workers: None,
        disp: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result = Shgo::new(expensive_schwefel, bounds)
        .with_options(options)
        .minimize();
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            println!("\nResult: f = {:.6e}", res.fun);
            println!("Location: x[0..5] = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, ...]", 
                res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]);
            println!("Function evaluations: {}", res.nfev);
            println!("Iterations: {}", res.nit);
            println!("Time: {:.3}s", elapsed.as_secs_f64());
            
            // Check how close we got
            let expected = 420.9687;
            let close_dims: usize = res.x.iter()
                .filter(|&&xi| (xi - expected).abs() < 50.0)
                .count();
            println!("\nDimensions within 50 of optimal (420.97): {}/10", close_dims);
            
            if res.fun < 100.0 {
                println!("SUCCESS: Found near-global minimum!");
            } else if res.fun < 500.0 {
                println!("CLOSE: Found good local minimum");
            } else {
                println!("MISS: Found inferior local minimum");
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
