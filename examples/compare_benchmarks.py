#!/usr/bin/env python3
"""
Run both Rust and Python SHGO benchmarks and compare results side-by-side.

Usage:
    python compare_benchmarks.py
    
Requirements:
    - Rust shgo-rs must be built: cargo build --release --examples
    - scipy must be installed
"""

import subprocess
import sys
import os
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent


def run_rust_benchmark():
    """Run the Rust benchmark and capture output."""
    print("Running Rust SHGO benchmark...")
    result = subprocess.run(
        ["cargo", "run", "--example", "benchmark_comparison", "--release"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Rust benchmark failed: {result.stderr}")
        return None
    return result.stdout


def run_python_benchmark():
    """Run the Python benchmark and capture output."""
    print("Running Python scipy.shgo benchmark...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "benchmark_scipy.py")],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Python benchmark failed: {result.stderr}")
        return None
    return result.stdout


def parse_csv_output(output):
    """Parse CSV output into a dictionary of results."""
    results = {}
    lines = output.strip().split('\n')
    
    # Find CSV data (skip header lines)
    csv_start = None
    for i, line in enumerate(lines):
        if line.startswith("function,dim,sampling"):
            csv_start = i + 1
            break
    
    if csv_start is None:
        return results
    
    for line in lines[csv_start:]:
        if not line or line.startswith("===") or line.startswith("sphere:") or line.startswith("rosenbrock:"):
            break
        
        parts = line.split(',')
        if len(parts) >= 9:
            # Normalize key (lowercase sampling method)
            key = f"{parts[0]}_{parts[1]}d_{parts[2].lower()}"
            results[key] = {
                'function': parts[0],
                'dim': int(parts[1]),
                'sampling': parts[2].lower(),
                'fun': float(parts[3]),
                'nfev': int(parts[4]),
                'nlfev': int(parts[5]),
                'nit': int(parts[6]),
                'time_ms': float(parts[7]),
                'success': parts[8].strip().lower() == 'true',
            }
    
    return results


def compare_results(rust_results, python_results):
    """Compare Rust and Python results."""
    print("\n" + "=" * 100)
    print("COMPARISON: Rust SHGO vs SciPy SHGO")
    print("=" * 100)
    print()
    
    # Header
    print(f"{'Test Case':<35} {'Rust f(x)':<15} {'SciPy f(x)':<15} {'Δf(x)':<15} {'Rust (ms)':<12} {'SciPy (ms)':<12} {'Speedup':<10}")
    print("-" * 100)
    
    total_rust_time = 0
    total_python_time = 0
    
    for key in sorted(rust_results.keys()):
        rust = rust_results.get(key)
        python = python_results.get(key)
        
        if rust and python:
            delta_fun = rust['fun'] - python['fun']
            speedup = python['time_ms'] / rust['time_ms'] if rust['time_ms'] > 0 else float('inf')
            
            total_rust_time += rust['time_ms']
            total_python_time += python['time_ms']
            
            # Format values
            rust_fun = f"{rust['fun']:.6e}"
            python_fun = f"{python['fun']:.6e}"
            delta_str = f"{delta_fun:+.2e}"
            rust_time = f"{rust['time_ms']:.2f}"
            python_time = f"{python['time_ms']:.2f}"
            speedup_str = f"{speedup:.1f}x"
            
            print(f"{key:<35} {rust_fun:<15} {python_fun:<15} {delta_str:<15} {rust_time:<12} {python_time:<12} {speedup_str:<10}")
    
    print("-" * 100)
    
    total_speedup = total_python_time / total_rust_time if total_rust_time > 0 else 0
    print(f"{'TOTAL':<35} {'':<15} {'':<15} {'':<15} {total_rust_time:<12.2f} {total_python_time:<12.2f} {total_speedup:.1f}x")
    print()
    
    # Summary statistics
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total Rust time:   {total_rust_time:.2f} ms")
    print(f"Total SciPy time:  {total_python_time:.2f} ms")
    print(f"Overall speedup:   {total_speedup:.1f}x")
    print()
    
    # Accuracy comparison
    print("Accuracy comparison (lower |Δf(x)| is better):")
    better_rust = 0
    better_python = 0
    equal = 0
    
    for key in rust_results.keys():
        rust = rust_results.get(key)
        python = python_results.get(key)
        if rust and python:
            # Compare to within 1e-6 tolerance
            if abs(rust['fun'] - python['fun']) < 1e-6:
                equal += 1
            elif rust['fun'] < python['fun']:
                better_rust += 1
            else:
                better_python += 1
    
    print(f"  Rust better:   {better_rust}")
    print(f"  SciPy better:  {better_python}")
    print(f"  Equal (±1e-6): {equal}")


def main():
    # Check if we're in the right directory
    if not (PROJECT_DIR / "Cargo.toml").exists():
        print("Error: Run this script from the shgo-rs project directory")
        sys.exit(1)
    
    # Run benchmarks
    rust_output = run_rust_benchmark()
    python_output = run_python_benchmark()
    
    if rust_output is None or python_output is None:
        print("Failed to run benchmarks")
        sys.exit(1)
    
    # Print raw outputs
    print("\n" + "=" * 50)
    print("RUST OUTPUT:")
    print("=" * 50)
    print(rust_output)
    
    print("\n" + "=" * 50)
    print("PYTHON OUTPUT:")
    print("=" * 50)
    print(python_output)
    
    # Parse and compare
    rust_results = parse_csv_output(rust_output)
    python_results = parse_csv_output(python_output)
    
    if rust_results and python_results:
        compare_results(rust_results, python_results)
    else:
        print("Failed to parse benchmark results")


if __name__ == "__main__":
    main()
