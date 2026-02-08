#!/usr/bin/env python3
"""
Comprehensive benchmark comparing scipy.optimize.shgo with the Rust implementation.

This script runs the same test functions as the Rust benchmark_comparison example
and outputs results in CSV format for easy comparison.

Usage:
    python benchmark_scipy.py
    
Then compare with:
    cargo run --example benchmark_comparison --release
"""

import numpy as np
from scipy.optimize import shgo
import time


def sphere(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2)


def rosenbrock(x):
    """Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x):
    """Rastrigin function"""
    a = 10.0
    n = len(x)
    return a * n + np.sum(x**2 - a * np.cos(2 * np.pi * x))


def ackley(x):
    """Ackley function"""
    n = len(x)
    a, b, c = 20.0, 0.2, 2 * np.pi
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e


def eggholder(x):
    """Eggholder function (2D only)"""
    x1, x2 = x[0], x[1]
    return (-(x2 + 47) * np.sin(np.sqrt(np.abs(x1/2 + x2 + 47))) 
            - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47)))))


def run_benchmark(name, func, bounds, sampling_method, maxiter, n=None):
    """Run a single benchmark and return results."""
    dim = len(bounds)
    
    options = {'maxiter': maxiter, 'disp': False}
    if n is not None:
        options['n'] = n
    
    start = time.perf_counter()
    result = shgo(func, bounds, sampling_method=sampling_method, options=options)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    return {
        'function': name,
        'dim': dim,
        'sampling': sampling_method,
        'x': result.x.tolist(),
        'fun': result.fun,
        'nfev': result.nfev,
        'nlfev': getattr(result, 'nlfev', 0),
        'nit': getattr(result, 'nit', 0),
        'time_ms': elapsed,
        'success': result.success,
    }


def print_result(r):
    """Print result in CSV format."""
    print(f"{r['function']},{r['dim']},{r['sampling']},{r['fun']:.10e},"
          f"{r['nfev']},{r['nlfev']},{r['nit']},{r['time_ms']:.4f},{r['success']}")


def main():
    print("=== SciPy SHGO Comprehensive Benchmark ===\n")
    
    # CSV header
    print("function,dim,sampling,fun,nfev,nlfev,nit,time_ms,success")
    
    # Sphere tests
    for dim in [2, 3, 5]:
        bounds = [(-5.0, 5.0)] * dim
        r = run_benchmark("sphere", sphere, bounds, "simplicial", 3)
        print_result(r)
        r = run_benchmark("sphere", sphere, bounds, "sobol", 3, n=64)
        print_result(r)
    
    # Rosenbrock tests
    for dim in [2, 3, 5]:
        bounds = [(-5.0, 5.0)] * dim
        r = run_benchmark("rosenbrock", rosenbrock, bounds, "simplicial", 3)
        print_result(r)
        r = run_benchmark("rosenbrock", rosenbrock, bounds, "sobol", 3, n=64)
        print_result(r)
    
    # Rastrigin tests
    for dim in [2, 3]:
        bounds = [(-5.12, 5.12)] * dim
        r = run_benchmark("rastrigin", rastrigin, bounds, "simplicial", 3)
        print_result(r)
        r = run_benchmark("rastrigin", rastrigin, bounds, "sobol", 3, n=64)
        print_result(r)
    
    # Ackley tests
    for dim in [2, 3]:
        bounds = [(-5.0, 5.0)] * dim
        r = run_benchmark("ackley", ackley, bounds, "simplicial", 3)
        print_result(r)
        r = run_benchmark("ackley", ackley, bounds, "sobol", 3, n=64)
        print_result(r)
    
    # Eggholder (2D only)
    bounds = [(-512.0, 512.0), (-512.0, 512.0)]
    r = run_benchmark("eggholder", eggholder, bounds, "simplicial", 5)
    print_result(r)
    r = run_benchmark("eggholder", eggholder, bounds, "sobol", 3, n=256)
    print_result(r)
    
    print("\n=== Expected Global Minima ===")
    print("sphere:     f(0, 0, ...) = 0")
    print("rosenbrock: f(1, 1, ...) = 0")
    print("rastrigin:  f(0, 0, ...) = 0")
    print("ackley:     f(0, 0, ...) = 0")
    print("eggholder:  f(512, 404.2319) ≈ -959.6407")


if __name__ == "__main__":
    main()
