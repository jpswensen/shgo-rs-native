#!/usr/bin/env python3
"""SciPy reference values for shgo-rs-native fidelity work (audit §4.3–4.5).

Mirrors the SciPy-comparable cases of examples/fix_benchmark.rs and emits the
same JSON-lines shape. Notes on comparability:
- SciPy's local minimizer is SLSQP (not BOBYQA), so nlfev and exact xl entries
  differ; fun/x/funl values and the SAMPLING structure (nit, sampling-eval
  counts, minimizer-pool sizes) are the fidelity targets.
- KNN/HNSW/ScaNN cases have no SciPy counterpart (documented Rust extension).
"""
import json
import time

import numpy as np
from scipy.optimize import shgo


def sphere(x):
    return float(np.sum(x * x))


def rastrigin(x):
    n = len(x)
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def report(name, secs, res):
    funl = np.atleast_1d(getattr(res, "funl", np.array([res.fun])))
    xl = np.atleast_2d(getattr(res, "xl", np.array([res.x])))
    print(json.dumps({
        "name": name,
        "secs": round(secs, 3),
        "fun": float(res.fun),
        "x": [float(v) for v in np.atleast_1d(res.x)],
        "funl_head": [float(v) for v in funl[:8]],
        "n_lmin": int(xl.shape[0]),
        "nfev": int(res.nfev),
        "nlfev": int(getattr(res, "nlfev", 0)),
        "nit": int(res.nit),
        "success": bool(res.success),
    }, sort_keys=True))


def run(name, **kw):
    t = time.time()
    res = shgo(**kw)
    report(name, time.time() - t, res)


# Delaunay path, matches sobol_delaunay_rastrigin_4d_1024x2
run(
    "scipy_sobol_rastrigin_4d_1024x2",
    func=rastrigin, bounds=[(-5.12, 5.12)] * 4,
    n=1024, sampling_method="sobol", options={"maxiter": 2},
)

# Smaller 6-D Delaunay case (8k points in 6-D would blow up qhull)
run(
    "scipy_sobol_rastrigin_6d_512x2",
    func=rastrigin, bounds=[(-5.12, 5.12)] * 6,
    n=512, sampling_method="sobol", options={"maxiter": 2},
)

# Simplicial defaults — the target for audit items 4.3/4.4/4.5
run(
    "scipy_simplicial_sphere_2d_default",
    func=sphere, bounds=[(-5.0, 5.0)] * 2, sampling_method="simplicial",
)
run(
    "scipy_simplicial_sphere_9d_default",
    func=sphere, bounds=[(-5.0, 5.0)] * 9, sampling_method="simplicial",
)
run(
    "scipy_simplicial_sphere_2d_maxiter3",
    func=sphere, bounds=[(-5.0, 5.0)] * 2, sampling_method="simplicial",
    options={"maxiter": 3},
)

# Constrained case, matches constrained_sphere_4d_1024x2
run(
    "scipy_constrained_sphere_4d_1024x2",
    func=sphere, bounds=[(-5.0, 5.0)] * 4,
    n=1024, sampling_method="sobol", options={"maxiter": 2},
    constraints=[{"type": "ineq", "fun": lambda x: x[0] + x[1] - 1.0}],
)
