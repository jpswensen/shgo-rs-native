//! Detailed per-component memory analysis for SHGO-RS.
//!
//! This example runs specific workloads and reports estimated memory usage
//! per major data structure component.
//!
//! Build: cargo build --example memory_breakdown --release --features track-alloc
//! Run:   ./target/release/examples/memory_breakdown

#[cfg(feature = "track-alloc")]
#[global_allocator]
static ALLOC: shgo::alloc_tracker::TrackingAllocator = shgo::alloc_tracker::TrackingAllocator::new();

use shgo::{Shgo, ShgoOptions, SamplingMethod};
use std::time::Instant;

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
        + 20.0
        + std::f64::consts::E
}

#[cfg(feature = "track-alloc")]
fn fmt_bytes(bytes: usize) -> String {
    shgo::alloc_tracker::AllocSnapshot::fmt_bytes(bytes)
}

#[cfg(feature = "track-alloc")]
struct PhaseMemory {
    name: &'static str,
    peak_bytes: usize,
    total_allocated: usize,
    alloc_count: usize,
    elapsed_ms: f64,
}

#[cfg(feature = "track-alloc")]
fn measure_phase<F: FnOnce()>(name: &'static str, f: F) -> PhaseMemory {
    shgo::alloc_tracker::reset_peak();
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    let snap = shgo::alloc_tracker::snapshot();
    PhaseMemory {
        name,
        peak_bytes: snap.peak_bytes,
        total_allocated: snap.total_allocated,
        alloc_count: snap.alloc_count,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

fn main() {
    #[cfg(not(feature = "track-alloc"))]
    {
        eprintln!("This example requires --features track-alloc");
        std::process::exit(1);
    }

    #[cfg(feature = "track-alloc")]
    {
        println!("══════════════════════════════════════════════════════════════════════");
        println!("  SHGO-RS Detailed Memory Breakdown");
        println!("══════════════════════════════════════════════════════════════════════");

        // Measure per-component sizes
        println!("\n── Estimated Struct Sizes ──────────────────────────────────────────");
        println!("  Coordinates(2D):  {} bytes (Vec<f64> + hash = 24 + 8 + 8 overhead = ~40B)",
            std::mem::size_of::<shgo::Coordinates>());
        println!("  Vertex:           {} bytes (coords + index + RwLocks + atomics)",
            std::mem::size_of::<shgo::Vertex>());
        println!("  ShgoOptions:      {} bytes",
            std::mem::size_of::<ShgoOptions>());

        println!("\n── Workload Memory Profiles ────────────────────────────────────────");

        let workloads: Vec<(&str, Box<dyn Fn()>)> = vec![
            ("Rosenbrock 2D Simp i=1", Box::new(|| {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                    .with_options(ShgoOptions { iters: Some(1), disp: 0, ..Default::default() })
                    .minimize();
            })),
            ("Rosenbrock 2D Simp i=3", Box::new(|| {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                    .with_options(ShgoOptions { iters: Some(3), disp: 0, ..Default::default() })
                    .minimize();
            })),
            ("Rosenbrock 2D Simp i=5", Box::new(|| {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                    .with_options(ShgoOptions { iters: Some(5), disp: 0, ..Default::default() })
                    .minimize();
            })),
            ("Rosenbrock 2D Simp i=7", Box::new(|| {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                    .with_options(ShgoOptions { iters: Some(7), disp: 0, ..Default::default() })
                    .minimize();
            })),
            ("Rastrigin 3D Sobol n=128", Box::new(|| {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); 3])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n: 128,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            })),
            ("Rastrigin 3D Sobol n=256", Box::new(|| {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); 3])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n: 256,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            })),
            ("Rastrigin 3D Sobol n=512", Box::new(|| {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); 3])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n: 512,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            })),
            ("Rastrigin 5D Sobol n=256", Box::new(|| {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); 5])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n: 256,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            })),
            ("Ackley 5D Sobol n=256", Box::new(|| {
                let _ = Shgo::new(ackley, vec![(-5.0, 5.0); 5])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n: 256,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            })),
            ("Rosenbrock 5D Simp i=1", Box::new(|| {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 5])
                    .with_options(ShgoOptions { iters: Some(1), disp: 0, ..Default::default() })
                    .minimize();
            })),
        ];

        // Warmup
        let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2]).minimize();

        println!();
        println!("  {:<30} {:>10} {:>12} {:>12} {:>10}",
            "Workload", "time(ms)", "peak_heap", "total_alloc", "allocs");
        println!("  {}", "─".repeat(78));

        for (name, func) in workloads {
            let m = measure_phase(name, func);
            println!("  {:<30} {:>10.2} {:>12} {:>12} {:>10}",
                m.name, m.elapsed_ms, fmt_bytes(m.peak_bytes),
                fmt_bytes(m.total_allocated), m.alloc_count);
        }

        // Memory scaling analysis
        println!("\n── Memory Scaling: Rosenbrock 2D Simplicial vs Iterations ────────");
        println!("  (Shows how memory grows with mesh refinement depth)");
        println!();
        println!("  {:<8} {:>10} {:>12} {:>12} {:>10} {:>8}",
            "iters", "time(ms)", "peak_heap", "total_alloc", "allocs", "alloc/B");

        for iters in 1..=7 {
            let m = measure_phase("", || {
                let _ = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                    .with_options(ShgoOptions { iters: Some(iters), disp: 0, ..Default::default() })
                    .minimize();
            });
            let avg_alloc_size = if m.alloc_count > 0 {
                m.total_allocated / m.alloc_count
            } else {
                0
            };
            println!("  {:<8} {:>10.2} {:>12} {:>12} {:>10} {:>8}",
                iters, m.elapsed_ms, fmt_bytes(m.peak_bytes),
                fmt_bytes(m.total_allocated), m.alloc_count, avg_alloc_size);
        }

        // Memory scaling: Sobol n
        println!("\n── Memory Scaling: Rastrigin 3D Sobol vs Sample Count ────────────");
        println!("  (Shows how memory grows with number of Sobol samples)");
        println!();
        println!("  {:<8} {:>10} {:>12} {:>12} {:>10} {:>8}",
            "n", "time(ms)", "peak_heap", "total_alloc", "allocs", "B/sample");

        for n in [32, 64, 128, 256, 512, 1024] {
            let m = measure_phase("", || {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); 3])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            });
            let bytes_per_sample = m.total_allocated / n.max(1);
            println!("  {:<8} {:>10.2} {:>12} {:>12} {:>10} {:>8}",
                n, m.elapsed_ms, fmt_bytes(m.peak_bytes),
                fmt_bytes(m.total_allocated), m.alloc_count, bytes_per_sample);
        }

        // Memory scaling: dimension
        println!("\n── Memory Scaling: Rastrigin Sobol n=128 vs Dimension ────────────");
        println!("  (Shows how memory grows with dimensionality)");
        println!();
        println!("  {:<8} {:>10} {:>12} {:>12} {:>10}",
            "dim", "time(ms)", "peak_heap", "total_alloc", "allocs");

        for dim in [2, 3, 4, 5, 6, 7] {
            let m = measure_phase("", || {
                let _ = Shgo::new(rastrigin, vec![(-5.12, 5.12); dim])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol, n: 128,
                        iters: Some(1), disp: 0, ..Default::default()
                    })
                    .minimize();
            });
            println!("  {:<8} {:>10.2} {:>12} {:>12} {:>10}",
                dim, m.elapsed_ms, fmt_bytes(m.peak_bytes),
                fmt_bytes(m.total_allocated), m.alloc_count);
        }

        println!();
    }
}
