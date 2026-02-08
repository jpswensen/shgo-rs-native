//! Integration tests using Python-generated fixtures for cross-validation.
//!
//! These tests ensure that the Rust implementation produces the same results
//! as the Python implementation.

use serde::Deserialize;
use shgo::{Complex, Coordinates, Sobol, VertexCache};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Get the path to the test fixtures directory.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

// ============================================================================
// Fixture Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct VertexTests {
    coordinates: CoordinateTests,
    midpoints: Vec<MidpointTest>,
}

#[derive(Debug, Deserialize)]
struct CoordinateTests {
    inputs: Vec<Vec<f64>>,
    hashes: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct MidpointTest {
    a: Vec<f64>,
    b: Vec<f64>,
    midpoint: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct CacheTests {
    field_evaluations: Vec<FieldEvaluation>,
}

#[derive(Debug, Deserialize)]
struct FieldEvaluation {
    coordinates: Vec<f64>,
    f_value: f64,
    expected_f: f64,
}

#[derive(Debug, Deserialize)]
struct TriangulationTests {
    triangulation_2d: Triangulation2D,
    triangulation_3d: Triangulation3D,
}

#[derive(Debug, Deserialize)]
struct Triangulation2D {
    bounds: Vec<Vec<f64>>,
    vertices: Vec<TriangulationVertex>,
    vertex_count: usize,
}

#[derive(Debug, Deserialize)]
struct Triangulation3D {
    bounds: Vec<Vec<f64>>,
    vertex_count: usize,
    expected_hypercube_vertices: usize,
}

#[derive(Debug, Deserialize)]
struct TriangulationVertex {
    coordinates: Vec<f64>,
    index: usize,
    f_value: f64,
    neighbors: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct MinimizerTests {
    star_pattern: StarPatternTest,
}

#[derive(Debug, Deserialize)]
struct StarPatternTest {
    center_coords: Vec<f64>,
    center_f: f64,
    center_is_minimizer: bool,
    neighbors: Vec<NeighborData>,
}

#[derive(Debug, Deserialize)]
struct NeighborData {
    coords: Vec<f64>,
    f: f64,
    is_minimizer: bool,
}

#[derive(Debug, Deserialize)]
struct SobolTests {
    sobol_2d: SobolTestCase,
    sobol_3d: SobolTestCase,
    sobol_5d: SobolTestCase,
}

#[derive(Debug, Deserialize)]
struct SobolTestCase {
    dim: usize,
    n: usize,
    skip: usize,
    points: Vec<Vec<f64>>,
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_midpoint_calculation_matches_python() {
    let fixture_path = fixtures_dir().join("vertex_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: VertexTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    for test in &tests.midpoints {
        let a = Coordinates::new(test.a.clone());
        let b = Coordinates::new(test.b.clone());
        let result = a.midpoint(&b);

        for (i, (&expected, &actual)) in test.midpoint.iter().zip(result.as_slice().iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-10,
                "Midpoint mismatch at index {}: expected {}, got {} (a={:?}, b={:?})",
                i, expected, actual, test.a, test.b
            );
        }
    }
}

#[test]
fn test_field_evaluation_matches_python() {
    let fixture_path = fixtures_dir().join("cache_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: CacheTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    // Use the same objective function as Python: sum of squares
    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

    // Create all vertices
    let vertices: Vec<_> = tests.field_evaluations
        .iter()
        .map(|test| cache.get_or_create(test.coordinates.clone()))
        .collect();

    // Process evaluations
    cache.process_pools();

    // Verify results match Python
    for (vertex, test) in vertices.iter().zip(tests.field_evaluations.iter()) {
        let f = vertex.f().expect("Vertex should be evaluated");
        
        assert!(
            (f - test.expected_f).abs() < 1e-10,
            "Field value mismatch at {:?}: expected {}, got {}",
            test.coordinates, test.expected_f, f
        );

        assert!(
            (f - test.f_value).abs() < 1e-10,
            "Field value doesn't match Python: expected {}, got {}",
            test.f_value, f
        );
    }
}

#[test]
fn test_minimizer_detection_matches_python() {
    let fixture_path = fixtures_dir().join("minimizer_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: MinimizerTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    let star = &tests.star_pattern;

    // Recreate the same star pattern
    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

    // Create center vertex
    let center = cache.get_or_create(star.center_coords.clone());

    // Create neighbor vertices
    let neighbors: Vec<_> = star.neighbors
        .iter()
        .map(|n| cache.get_or_create(n.coords.clone()))
        .collect();

    // Connect center to all neighbors (bidirectional)
    for neighbor in &neighbors {
        center.connect(neighbor.index());
        neighbor.connect(center.index());
    }

    // Evaluate
    cache.process_pools();

    // Verify center function value
    let center_f = center.f().expect("Center should be evaluated");
    assert!(
        (center_f - star.center_f).abs() < 1e-10,
        "Center f value mismatch: expected {}, got {}",
        star.center_f, center_f
    );

    // Verify center is minimizer
    let center_is_min = cache.is_minimizer(&center);
    assert_eq!(
        center_is_min, star.center_is_minimizer,
        "Center minimizer status mismatch: expected {}, got {}",
        star.center_is_minimizer, center_is_min
    );

    // Verify neighbor function values and minimizer status
    for (neighbor, expected) in neighbors.iter().zip(star.neighbors.iter()) {
        let f = neighbor.f().expect("Neighbor should be evaluated");
        assert!(
            (f - expected.f).abs() < 1e-10,
            "Neighbor f value mismatch at {:?}: expected {}, got {}",
            expected.coords, expected.f, f
        );

        let is_min = cache.is_minimizer(neighbor);
        assert_eq!(
            is_min, expected.is_minimizer,
            "Neighbor minimizer status mismatch at {:?}: expected {}, got {}",
            expected.coords, expected.is_minimizer, is_min
        );
    }
}

#[test]
fn test_triangulation_vertex_count() {
    let fixture_path = fixtures_dir().join("triangulation_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: TriangulationTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    // Verify 2D triangulation expectations
    let tri_2d = &tests.triangulation_2d;
    println!("2D triangulation: {} vertices", tri_2d.vertex_count);
    println!("Expected: 4 corners + 1 centroid = 5 vertices");
    assert_eq!(tri_2d.vertex_count, 5, "2D triangulation should have 5 vertices");

    // Verify 3D triangulation expectations
    let tri_3d = &tests.triangulation_3d;
    println!("3D triangulation: {} vertices", tri_3d.vertex_count);
    println!("Expected hypercube vertices: {}", tri_3d.expected_hypercube_vertices);
    assert!(
        tri_3d.vertex_count >= tri_3d.expected_hypercube_vertices,
        "3D triangulation should have at least {} hypercube vertices, got {}",
        tri_3d.expected_hypercube_vertices, tri_3d.vertex_count
    );
}

#[test]
fn test_triangulation_connectivity_2d() {
    let fixture_path = fixtures_dir().join("triangulation_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: TriangulationTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    let tri = &tests.triangulation_2d;

    // Recreate the triangulation structure in Rust
    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

    // Create vertices with same coordinates as Python
    let mut index_to_vertex = HashMap::new();
    for v in &tri.vertices {
        let vertex = cache.get_or_create(v.coordinates.clone());
        index_to_vertex.insert(v.index, vertex);
    }

    // Add connections as in Python
    for v in &tri.vertices {
        let vertex = index_to_vertex.get(&v.index).unwrap();
        for &neighbor_idx in &v.neighbors {
            vertex.connect(neighbor_idx);
        }
    }

    // Evaluate
    cache.process_pools();

    // Verify function values match
    for v in &tri.vertices {
        let vertex = index_to_vertex.get(&v.index).unwrap();
        let f = vertex.f().expect("Vertex should be evaluated");
        assert!(
            (f - v.f_value).abs() < 1e-10,
            "Function value mismatch at {:?}: expected {}, got {}",
            v.coordinates, v.f_value, f
        );
    }

    // Find minimizers - should find the origin (0,0) with f=0
    let minimizers = cache.find_all_minimizers();
    
    // The origin should be a minimizer
    let origin_is_minimizer = minimizers.iter().any(|v| {
        v.x().iter().all(|&x| x.abs() < 1e-10)
    });
    
    assert!(
        origin_is_minimizer,
        "Origin should be a minimizer in the 2D triangulation"
    );
}

#[test]
fn test_concurrent_cache_operations() {
    // This test verifies thread safety matches expected behavior
    use std::sync::Arc;
    use std::thread;

    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let cache: Arc<VertexCache<_, fn(&[f64]) -> bool>> = 
        Arc::new(VertexCache::new(objective, None));

    // Spawn threads that create vertices concurrently
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                for i in 0..100 {
                    let x = (t * 100 + i) as f64 / 400.0;
                    cache.get_or_create(vec![x, x]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have 400 unique vertices
    assert_eq!(cache.len(), 400);

    // Process all at once
    cache.process_pools();

    // Verify all are evaluated
    assert_eq!(cache.function_evaluations(), 400);
}

#[test]
fn test_complex_triangulation_matches_python_vertex_count() {
    let fixture_path = fixtures_dir().join("triangulation_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: TriangulationTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    // Test 2D Complex triangulation
    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let bounds_2d = vec![(0.0, 1.0), (0.0, 1.0)];
    let mut complex_2d: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds_2d, objective, None);
    
    complex_2d.triangulate(None, true);
    
    // Verify vertex count matches Python
    assert_eq!(
        complex_2d.vertex_count(), tests.triangulation_2d.vertex_count,
        "2D Complex vertex count mismatch: expected {}, got {}",
        tests.triangulation_2d.vertex_count, complex_2d.vertex_count()
    );

    // Test 3D Complex triangulation
    let bounds_3d = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let mut complex_3d: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds_3d, objective, None);
    
    complex_3d.triangulate(None, true);
    
    // 3D should have at least the expected hypercube vertices
    assert!(
        complex_3d.vertex_count() >= tests.triangulation_3d.expected_hypercube_vertices,
        "3D Complex should have at least {} vertices, got {}",
        tests.triangulation_3d.expected_hypercube_vertices, complex_3d.vertex_count()
    );
}

#[test]
fn test_complex_field_evaluation_matches_python() {
    let fixture_path = fixtures_dir().join("cache_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: CacheTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    // Use Complex instead of raw VertexCache
    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let bounds = vec![(0.0, 10.0), (0.0, 10.0)]; // Wide enough for test points
    let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, objective, None);
    
    // Don't triangulate - just use the cache directly for this test
    // Create vertices for each test point
    let vertices: Vec<_> = tests.field_evaluations
        .iter()
        .map(|test| complex.cache.get_or_create(test.coordinates.clone()))
        .collect();

    // Process evaluations
    complex.process_pools();

    // Verify results match Python
    for (vertex, test) in vertices.iter().zip(tests.field_evaluations.iter()) {
        let f = vertex.f().expect("Vertex should be evaluated");
        
        assert!(
            (f - test.expected_f).abs() < 1e-10,
            "Field value mismatch at {:?}: expected {}, got {}",
            test.coordinates, test.expected_f, f
        );
    }
}

#[test]
fn test_complex_minimizer_finding() {
    // Test that Complex.find_minimizers works correctly
    let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, objective, None);
    
    complex.triangulate(None, true);
    complex.process_pools();
    
    let minimizers = complex.find_minimizers();
    
    // For sphere function on [0,1]^2, origin (0,0) should be a minimizer
    assert!(!minimizers.is_empty(), "Should find at least one minimizer");
    
    let has_origin = minimizers.iter().any(|v| {
        v.x().iter().all(|&x| x.abs() < 1e-10)
    });
    
    assert!(has_origin, "Origin (0,0) should be the minimizer for sphere function");
    
    // The minimizer at origin should have f=0
    let origin_min = minimizers.iter().find(|v| {
        v.x().iter().all(|&x| x.abs() < 1e-10)
    }).unwrap();
    
    assert!(
        origin_min.f().unwrap().abs() < 1e-10,
        "Minimum value should be 0 at origin"
    );
}

#[test]
fn test_sobol_sequence_matches_python() {
    let fixture_path = fixtures_dir().join("sobol_tests.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", fixture_path, e));
    let tests: SobolTests = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e));

    // Test 2D Sobol sequence
    let test_2d = &tests.sobol_2d;
    let mut sobol_2d = Sobol::new(test_2d.dim);
    let points_2d = sobol_2d.generate(test_2d.n, test_2d.skip);
    
    assert_eq!(points_2d.len(), test_2d.points.len());
    for (i, (rust_point, python_point)) in points_2d.iter().zip(test_2d.points.iter()).enumerate() {
        for (j, (&rust_val, &python_val)) in rust_point.iter().zip(python_point.iter()).enumerate() {
            assert!(
                (rust_val - python_val).abs() < 1e-10,
                "2D Sobol mismatch at point {}, dim {}: Rust={}, Python={}",
                i, j, rust_val, python_val
            );
        }
    }

    // Test 3D Sobol sequence
    let test_3d = &tests.sobol_3d;
    let mut sobol_3d = Sobol::new(test_3d.dim);
    let points_3d = sobol_3d.generate(test_3d.n, test_3d.skip);
    
    assert_eq!(points_3d.len(), test_3d.points.len());
    for (i, (rust_point, python_point)) in points_3d.iter().zip(test_3d.points.iter()).enumerate() {
        for (j, (&rust_val, &python_val)) in rust_point.iter().zip(python_point.iter()).enumerate() {
            assert!(
                (rust_val - python_val).abs() < 1e-10,
                "3D Sobol mismatch at point {}, dim {}: Rust={}, Python={}",
                i, j, rust_val, python_val
            );
        }
    }

    // Test 5D Sobol sequence
    let test_5d = &tests.sobol_5d;
    let mut sobol_5d = Sobol::new(test_5d.dim);
    let points_5d = sobol_5d.generate(test_5d.n, test_5d.skip);
    
    assert_eq!(points_5d.len(), test_5d.points.len());
    for (i, (rust_point, python_point)) in points_5d.iter().zip(test_5d.points.iter()).enumerate() {
        for (j, (&rust_val, &python_val)) in rust_point.iter().zip(python_point.iter()).enumerate() {
            assert!(
                (rust_val - python_val).abs() < 1e-10,
                "5D Sobol mismatch at point {}, dim {}: Rust={}, Python={}",
                i, j, rust_val, python_val
            );
        }
    }
}
