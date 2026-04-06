//! Flamegraph profiling target for SHGO-RS.
//!
//! Run specific workloads for profiling with cargo-flamegraph or dtrace.
//!
//! Usage:
//!   cargo flamegraph --example flamegraph_target --release -- <workload>
//!
//! Workloads:
//!   rosenbrock_2d_simp_i5   - 2D Rosenbrock Simplicial iters=5 (deep mesh)
//!   rastrigin_5d_sobol      - 5D Rastrigin Sobol n=256 (local opt heavy)
//!   ackley_5d_sobol         - 5D Ackley Sobol n=256
//!   rosenbrock_5d_simp      - 5D Rosenbrock Simplicial (triangulation heavy)
//!   eggholder_2d_simp       - 2D Eggholder Simplicial iters=3

use shgo::{Shgo, ShgoOptions, SamplingMethod};

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workload = args.get(1).map(|s| s.as_str()).unwrap_or("rosenbrock_2d_simp_i5");

    // Warmup
    let _ = Shgo::new(sphere, vec![(-5.0, 5.0); 2]).minimize();

    let n_repeats = match workload {
        "rosenbrock_2d_simp_i5" => 20,
        _ => 10,
    };

    eprintln!("Running workload '{}' x{} repeats...", workload, n_repeats);

    for _ in 0..n_repeats {
        match workload {
            "rosenbrock_2d_simp_i5" => {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                    .with_options(ShgoOptions {
                        iters: Some(5),
                        disp: 0,
                        ..Default::default()
                    })
                    .minimize();
            }
            "rastrigin_5d_sobol" => {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); 5])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol,
                        n: 256,
                        iters: Some(1),
                        disp: 0,
                        ..Default::default()
                    })
                    .minimize();
            }
            "ackley_5d_sobol" => {
                let _ = Shgo::new(ackley, vec![(-5.0, 5.0); 5])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol,
                        n: 256,
                        iters: Some(1),
                        disp: 0,
                        ..Default::default()
                    })
                    .minimize();
            }
            "rosenbrock_5d_simp" => {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 5])
                    .with_options(ShgoOptions {
                        iters: Some(1),
                        disp: 0,
                        ..Default::default()
                    })
                    .minimize();
            }
            "eggholder_2d_simp" => {
                let _ = Shgo::new(eggholder, vec![(-512.0, 512.0); 2])
                    .with_options(ShgoOptions {
                        iters: Some(3),
                        disp: 0,
                        ..Default::default()
                    })
                    .minimize();
            }
            _ => {
                eprintln!("Unknown workload: {}", workload);
                eprintln!("Available: rosenbrock_2d_simp_i5, rastrigin_5d_sobol, ackley_5d_sobol, rosenbrock_5d_simp, eggholder_2d_simp");
                std::process::exit(1);
            }
        }
    }
    eprintln!("Done.");
}
