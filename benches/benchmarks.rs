//! Benchmarks for SHGO-RS
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use shgo::{Coordinates, Vertex, VertexCache};
use std::sync::Arc;

fn benchmark_coordinates(c: &mut Criterion) {
    let mut group = c.benchmark_group("Coordinates");

    // Benchmark coordinate creation
    for dim in [2, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("new", dim),
            dim,
            |b, &dim| {
                let values: Vec<f64> = (0..dim).map(|i| i as f64).collect();
                b.iter(|| Coordinates::new(black_box(values.clone())))
            },
        );
    }

    // Benchmark midpoint calculation
    group.bench_function("midpoint_3d", |b| {
        let a = Coordinates::new(vec![0.0, 0.0, 0.0]);
        let c = Coordinates::new(vec![1.0, 1.0, 1.0]);
        b.iter(|| a.midpoint(black_box(&c)))
    });

    // Benchmark hash lookup (simulating cache access)
    group.bench_function("hash_lookup", |b| {
        use std::collections::HashMap;
        let mut map: HashMap<Coordinates, i32> = HashMap::new();
        for i in 0..1000 {
            map.insert(Coordinates::new(vec![i as f64, 0.0]), i);
        }
        let key = Coordinates::new(vec![500.0, 0.0]);
        b.iter(|| map.get(black_box(&key)))
    });

    group.finish();
}

fn benchmark_vertex(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vertex");

    // Benchmark vertex creation
    group.bench_function("new_3d", |b| {
        b.iter(|| Vertex::new(black_box(vec![1.0, 2.0, 3.0]), 0))
    });

    // Benchmark connection operations
    group.bench_function("connect_100", |b| {
        let v = Vertex::new(vec![0.0, 0.0], 0);
        b.iter(|| {
            for i in 1..=100 {
                v.connect(black_box(i));
            }
        })
    });

    group.finish();
}

fn benchmark_vertex_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("VertexCache");

    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();

    // Benchmark vertex creation
    group.bench_function("get_or_create_1000", |b| {
        b.iter(|| {
            let cache: VertexCache<_, fn(&[f64]) -> bool> = 
                VertexCache::new(objective, None);
            for i in 0..1000 {
                cache.get_or_create(vec![i as f64 / 1000.0, 0.0]);
            }
        })
    });

    // Benchmark batch evaluation
    group.bench_function("process_pools_1000", |b| {
        b.iter_batched(
            || {
                let cache: VertexCache<_, fn(&[f64]) -> bool> = 
                    VertexCache::new(objective, None);
                for i in 0..1000 {
                    cache.get_or_create(vec![i as f64 / 1000.0, 0.0]);
                }
                cache
            },
            |cache| cache.process_pools(),
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark minimizer finding
    group.bench_function("find_minimizers_grid_100", |b| {
        b.iter_batched(
            || {
                let cache: VertexCache<_, fn(&[f64]) -> bool> = 
                    VertexCache::new(objective, None);
                
                // Create a 10x10 grid
                let mut vertices = Vec::new();
                for i in 0..10 {
                    for j in 0..10 {
                        let v = cache.get_or_create(vec![i as f64, j as f64]);
                        vertices.push(v);
                    }
                }
                
                // Connect neighbors
                for i in 0..10 {
                    for j in 0..10 {
                        let idx = i * 10 + j;
                        if i > 0 {
                            Vertex::connect_bidirectional(&vertices[idx], &vertices[idx - 10]);
                        }
                        if j > 0 {
                            Vertex::connect_bidirectional(&vertices[idx], &vertices[idx - 1]);
                        }
                    }
                }
                
                cache.process_pools();
                cache
            },
            |cache| cache.find_all_minimizers(),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ─── Full SHGO Optimization Benchmarks ──────────────────────────────────────

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

fn benchmark_shgo_minimize(c: &mut Criterion) {
    use shgo::{Shgo, ShgoOptions, SamplingMethod, ConnectivityMethod};

    let mut group = c.benchmark_group("SHGO_minimize");
    // Increase measurement time for longer-running benchmarks
    group.sample_size(10);

    // 2D Sphere - Simplicial (fastest baseline)
    group.bench_function("sphere_2d_simp", |b| {
        b.iter(|| {
            let result = Shgo::new(sphere, vec![(-5.0, 5.0); 2])
                .with_options(ShgoOptions {
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 2D Rosenbrock - Simplicial
    group.bench_function("rosenbrock_2d_simp", |b| {
        b.iter(|| {
            let result = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                .with_options(ShgoOptions {
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 2D Rastrigin - Sobol n=128
    group.bench_function("rastrigin_2d_sobol_n128", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 2])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    n: 128,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 3D Rosenbrock - Simplicial
    group.bench_function("rosenbrock_3d_simp", |b| {
        b.iter(|| {
            let result = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 3])
                .with_options(ShgoOptions {
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 5D Ackley - Sobol n=256
    group.bench_function("ackley_5d_sobol_n256", |b| {
        b.iter(|| {
            let result = Shgo::new(ackley, vec![(-5.0, 5.0); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    n: 256,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 5D Rastrigin - Sobol n=512 (heavier)
    group.bench_function("rastrigin_5d_sobol_n512", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    n: 512,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 2D Rosenbrock - deep refinement (iters=5)
    group.bench_function("rosenbrock_2d_simp_i5", |b| {
        b.iter(|| {
            let result = Shgo::new(rosenbrock, vec![(-5.0, 5.0); 2])
                .with_options(ShgoOptions {
                    iters: Some(5),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // ---- k-NN connectivity benchmarks (direct comparison with Delaunay) ----

    // 2D Rastrigin - Sobol n=128 + k-NN
    group.bench_function("rastrigin_2d_sobol_n128_knn", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 2])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::KNearestNeighbors,
                    n: 128,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 5D Ackley - Sobol n=256 + k-NN
    group.bench_function("ackley_5d_sobol_n256_knn", |b| {
        b.iter(|| {
            let result = Shgo::new(ackley, vec![(-5.0, 5.0); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::KNearestNeighbors,
                    n: 256,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // 5D Rastrigin - Sobol n=512 + k-NN
    group.bench_function("rastrigin_5d_sobol_n512_knn", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::KNearestNeighbors,
                    n: 512,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // ---- HNSW connectivity benchmarks ----

    group.bench_function("rastrigin_2d_sobol_n128_hnsw", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 2])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::HNSW,
                    n: 128,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    group.bench_function("ackley_5d_sobol_n256_hnsw", |b| {
        b.iter(|| {
            let result = Shgo::new(ackley, vec![(-5.0, 5.0); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::HNSW,
                    n: 256,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    group.bench_function("rastrigin_5d_sobol_n512_hnsw", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::HNSW,
                    n: 512,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    // ---- ScaNN connectivity benchmarks ----

    group.bench_function("rastrigin_2d_sobol_n128_scann", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 2])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::ScaNN,
                    n: 128,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    group.bench_function("ackley_5d_sobol_n256_scann", |b| {
        b.iter(|| {
            let result = Shgo::new(ackley, vec![(-5.0, 5.0); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::ScaNN,
                    n: 256,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    group.bench_function("rastrigin_5d_sobol_n512_scann", |b| {
        b.iter(|| {
            let result = Shgo::new(rastrigin, vec![(-5.12, 5.12); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::ScaNN,
                    n: 512,
                    iters: Some(1),
                    disp: 0,
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_coordinates,
    benchmark_vertex,
    benchmark_vertex_cache,
    benchmark_shgo_minimize
);
criterion_main!(benches);
