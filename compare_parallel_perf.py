#!/usr/bin/env python3
"""Compare v1_nonparallel vs parallelized benchmark results."""
import json
import math

# Load new parallel results
with open("rust_perf_results.json") as f:
    new = {r["name"]: r for r in json.load(f)}

# Old v1_nonparallel medians from benchmark run (in microseconds)
old = {
    "sphere_2d_simp": 338,
    "sphere_2d_sobol": 960,
    "rosenbrock_2d_simp": 452,
    "rosenbrock_2d_sobol": 1677,
    "ackley_2d_simp": 354,
    "rastrigin_2d_simp": 337,
    "rastrigin_2d_sobol": 1568,
    "eggholder_2d_simp": 2260,
    "eggholder_2d_sobol": 3514,
    "styblinski_2d_simp": 366,
    "levy_2d_simp": 396,
    "sphere_5d_simp": 2956,
    "rosenbrock_3d_simp": 776,
    "sphere_2d_constrained": 433,
    "rastrigin_3d_sobol": 2750,
    "rosenbrock_5d_sobol": 136000,
    "ackley_5d_sobol": 94000,
    "rastrigin_5d_sobol": 94000,
}

print(f"{'Test':<30} {'Old(us)':>10} {'New(us)':>10} {'Speedup':>10}")
print("-" * 65)
log_speedups = []
for name in sorted(old.keys()):
    o = old[name]
    if name in new:
        n = new[name]["median_us"]
        sp = o / n
        log_speedups.append(math.log(sp))
        print(f"{name:<30} {o:>10.0f} {n:>10.0f} {sp:>10.2f}x")
    else:
        print(f"{name:<30} {o:>10.0f} {'N/A':>10}")

geo_mean = math.exp(sum(log_speedups) / len(log_speedups))
print(f"\nGeometric mean parallel speedup: {geo_mean:.2f}x")
print(f"Tests compared: {len(log_speedups)}")
