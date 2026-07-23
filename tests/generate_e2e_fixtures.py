#!/usr/bin/env python3
"""Generate end-to-end SHGO fixtures from SciPy for cross-validation.

Runs scipy.optimize.shgo on cases where the found optimum is robust to the
local-minimizer difference (SciPy uses SLSQP, the Rust port uses BOBYQA), and
records the final fun/x. The Rust integration test
(`test_end_to_end_matches_scipy` in tests/cross_validation.rs) replays the
same configurations and asserts agreement within tolerance.

Usage (needs scipy — e.g. `python3 -m venv .venv && .venv/bin/pip install
scipy numpy`, then run with that interpreter from the repo root):

    .venv/bin/python tests/generate_e2e_fixtures.py
"""
import json
import os

import numpy as np
from scipy.optimize import shgo


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


FUNCS = {"sphere": sphere, "rosenbrock": rosenbrock}

CASES = [
    # name, func, bounds, n (0 = scipy default/auto), iters, method, constrained
    ("e2e_sphere_2d_simplicial_default", "sphere", [(-5.0, 5.0)] * 2, 0, 1,
     "simplicial", False),
    ("e2e_sphere_4d_simplicial_iters2", "sphere", [(-5.0, 5.0)] * 4, 0, 2,
     "simplicial", False),
    ("e2e_sphere_3d_sobol_64", "sphere", [(-5.0, 5.0)] * 3, 64, 1,
     "sobol", False),
    ("e2e_rosenbrock_2d_sobol_128", "rosenbrock", [(-2.0, 2.0)] * 2, 128, 1,
     "sobol", False),
    ("e2e_constrained_sphere_2d_sobol_64", "sphere", [(-5.0, 5.0)] * 2, 64, 1,
     "sobol", True),
]


def main():
    out = {"scipy_version": __import__("scipy").__version__, "cases": []}
    for name, fname, bounds, n, iters, method, constrained in CASES:
        kwargs = dict(func=FUNCS[fname], bounds=bounds, iters=iters,
                      sampling_method=method)
        if n:
            kwargs["n"] = n
        if constrained:
            kwargs["constraints"] = [
                {"type": "ineq", "fun": lambda x: x[0] + x[1] - 1.0}
            ]
        res = shgo(**kwargs)
        assert res.success, f"{name}: scipy run failed: {res.message}"
        out["cases"].append({
            "name": name,
            "func": fname,
            "bounds": [list(b) for b in bounds],
            "n": n,
            "iters": iters,
            "sampling_method": method,
            "constrained": constrained,
            "fun": float(res.fun),
            "x": [float(v) for v in np.atleast_1d(res.x)],
        })
        print(f"{name}: fun={res.fun:.6e} x={np.round(res.x, 6).tolist()}")

    path = os.path.join(os.path.dirname(__file__), "fixtures",
                        "end_to_end_tests.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
