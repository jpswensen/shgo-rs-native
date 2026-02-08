#!/usr/bin/env python3
"""
Generate test fixtures for cross-validation between Python and Rust implementations.

This script creates JSON files containing expected values from the Python SHGO
implementation that can be used to validate the Rust implementation.
"""

import json
import sys
import os

# Add the shgo module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shgo'))

import numpy as np
from shgo._shgo_lib._vertex import VertexScalarField, VertexCacheField
from shgo._shgo_lib._complex import Complex


def generate_vertex_tests():
    """Generate test data for vertex operations."""
    tests = {}
    
    # Test basic vertex creation and hashing
    test_coordinates = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
    ]
    
    tests['coordinates'] = {
        'inputs': test_coordinates,
        'hashes': [hash(tuple(c)) for c in test_coordinates],
    }
    
    # Test midpoint calculation
    midpoint_tests = []
    for i, c1 in enumerate(test_coordinates):
        for c2 in test_coordinates[i+1:]:
            midpoint = [(a + b) / 2.0 for a, b in zip(c1, c2)]
            midpoint_tests.append({
                'a': c1,
                'b': c2,
                'midpoint': midpoint
            })
    tests['midpoints'] = midpoint_tests
    
    return tests


def generate_cache_tests():
    """Generate test data for vertex cache operations."""
    tests = {}
    
    # Simple quadratic function
    def objective(x):
        return sum(xi**2 for xi in x)
    
    # Create cache and add vertices
    cache = VertexCacheField(field=objective, field_args=())
    
    # Add some test vertices
    test_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.25, 0.75),
        (-1.0, -1.0),
        (2.0, 3.0),
    ]
    
    vertices = []
    for pt in test_points:
        v = cache[pt]
        vertices.append(v)
    
    # Process the pool
    cache.process_pools()
    
    # Collect results
    tests['field_evaluations'] = [
        {
            'coordinates': list(v.x),
            'f_value': v.f,
            'expected_f': sum(xi**2 for xi in v.x),
        }
        for v in vertices
    ]
    
    return tests


def generate_triangulation_tests():
    """Generate test data for triangulation operations."""
    tests = {}
    
    # Simple 2D case
    def objective_2d(x):
        return x[0]**2 + x[1]**2
    
    bounds_2d = [(0.0, 1.0), (0.0, 1.0)]
    complex_2d = Complex(
        dim=2,
        domain=bounds_2d,
        sfield=objective_2d,
        sfield_args=(),
    )
    
    # Perform initial triangulation
    complex_2d.triangulate(centroid=True)
    complex_2d.V.process_pools()
    
    # Collect vertices
    vertices_2d = []
    for coords, vertex in complex_2d.V.cache.items():
        vertices_2d.append({
            'coordinates': list(coords),
            'index': vertex.index,
            'f_value': vertex.f if hasattr(vertex, 'f') else None,
            'neighbors': [complex_2d.V.cache[tuple(n.x)].index 
                         for n in vertex.nn if tuple(n.x) in complex_2d.V.cache],
        })
    
    tests['triangulation_2d'] = {
        'bounds': bounds_2d,
        'vertices': vertices_2d,
        'vertex_count': len(vertices_2d),
    }
    
    # 3D case
    def objective_3d(x):
        return x[0]**2 + x[1]**2 + x[2]**2
    
    bounds_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    complex_3d = Complex(
        dim=3,
        domain=bounds_3d,
        sfield=objective_3d,
        sfield_args=(),
    )
    
    complex_3d.triangulate(centroid=True)
    complex_3d.V.process_pools()
    
    vertices_3d = []
    for coords, vertex in complex_3d.V.cache.items():
        vertices_3d.append({
            'coordinates': list(coords),
            'index': vertex.index,
        })
    
    tests['triangulation_3d'] = {
        'bounds': bounds_3d,
        'vertex_count': len(vertices_3d),
        'expected_hypercube_vertices': 2**3,  # 8 corners
    }
    
    return tests


def generate_minimizer_tests():
    """Generate test data for minimizer detection."""
    tests = {}
    
    # Create a function with a known minimum at origin
    def objective(x):
        return x[0]**2 + x[1]**2
    
    cache = VertexCacheField(field=objective, field_args=())
    
    # Create vertices forming a star pattern around origin
    center = cache[(0.0, 0.0)]
    v1 = cache[(1.0, 0.0)]
    v2 = cache[(0.0, 1.0)]
    v3 = cache[(-1.0, 0.0)]
    v4 = cache[(0.0, -1.0)]
    
    # Connect center to all neighbors
    center.connect(v1)
    center.connect(v2)
    center.connect(v3)
    center.connect(v4)
    
    # Process evaluations
    cache.process_pools()
    
    # Check minimizers
    tests['star_pattern'] = {
        'center_coords': [0.0, 0.0],
        'center_f': center.f,
        'center_is_minimizer': center.minimiser(),
        'neighbors': [
            {'coords': list(v.x), 'f': v.f, 'is_minimizer': v.minimiser()}
            for v in [v1, v2, v3, v4]
        ]
    }
    
    return tests


def generate_sobol_tests():
    """Generate test data for Sobol sequence validation.
    
    Uses scipy.stats.qmc.Sobol with Joe-Kuo (2008) direction numbers,
    which is the same Sobol engine used by SHGO in modern scipy.
    """
    from scipy.stats import qmc
    
    tests = {}
    
    for label, dim, n, skip in [('sobol_2d', 2, 20, 1),
                                 ('sobol_3d', 3, 15, 1),
                                 ('sobol_5d', 5, 10, 1)]:
        sobol = qmc.Sobol(d=dim, scramble=False, seed=0)
        # Generate skip+n points, then discard the first 'skip'
        all_points = sobol.random(skip + n)
        points = all_points[skip:]
        tests[label] = {
            'dim': dim,
            'n': n,
            'skip': skip,
            'points': points.tolist()
        }
    
    return tests


def main():
    """Generate all test fixtures."""
    output_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save vertex tests
    print("Generating vertex tests...")
    vertex_tests = generate_vertex_tests()
    with open(os.path.join(output_dir, 'vertex_tests.json'), 'w') as f:
        json.dump(vertex_tests, f, indent=2)
    print(f"  Saved {len(vertex_tests)} test categories")
    
    # Generate and save cache tests
    print("Generating cache tests...")
    cache_tests = generate_cache_tests()
    with open(os.path.join(output_dir, 'cache_tests.json'), 'w') as f:
        json.dump(cache_tests, f, indent=2)
    print(f"  Saved {len(cache_tests['field_evaluations'])} field evaluation tests")
    
    # Generate and save triangulation tests
    print("Generating triangulation tests...")
    tri_tests = generate_triangulation_tests()
    with open(os.path.join(output_dir, 'triangulation_tests.json'), 'w') as f:
        json.dump(tri_tests, f, indent=2)
    print(f"  Saved 2D ({tri_tests['triangulation_2d']['vertex_count']} vertices) "
          f"and 3D ({tri_tests['triangulation_3d']['vertex_count']} vertices) tests")
    
    # Generate and save minimizer tests
    print("Generating minimizer tests...")
    min_tests = generate_minimizer_tests()
    with open(os.path.join(output_dir, 'minimizer_tests.json'), 'w') as f:
        json.dump(min_tests, f, indent=2)
    print(f"  Saved minimizer detection tests")
    
    # Generate and save Sobol tests
    print("Generating Sobol sequence tests...")
    sobol_tests = generate_sobol_tests()
    with open(os.path.join(output_dir, 'sobol_tests.json'), 'w') as f:
        json.dump(sobol_tests, f, indent=2)
    print(f"  Saved Sobol sequence tests (2D: {sobol_tests['sobol_2d']['n']}, "
          f"3D: {sobol_tests['sobol_3d']['n']}, 5D: {sobol_tests['sobol_5d']['n']} points)")
    
    print("\nAll test fixtures generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
