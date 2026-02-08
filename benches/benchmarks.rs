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

criterion_group!(
    benches,
    benchmark_coordinates,
    benchmark_vertex,
    benchmark_vertex_cache
);
criterion_main!(benches);
