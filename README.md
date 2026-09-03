# shgo-rs

A high-performance, faithful Rust-native implementation of the
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html)
SHGO (Simplicial Homology Global Optimization) algorithm, with optional
[rayon](https://docs.rs/rayon) parallelism for local minimization.

## Overview

SHGO is a global optimization algorithm that uses concepts from algebraic
topology — specifically simplicial homology — to systematically identify *all*
local minima of a function over a bounded domain and return the global minimum
among them. Key properties:

- **Derivative-free** — no gradients, smoothness, or convexity required.
- **Theoretically convergent** — guaranteed to find the global minimum given
  sufficient sampling.
- **All-minima discovery** — returns the complete set of local minima found at
  each iteration, not just the best one.
- **Two sampling modes** — *Simplicial* (default, topology-aware) and *Sobol*
  (quasi-random, higher dimensional).

This crate is a port of the SciPy SHGO implementation with its deviations and
extensions documented (see `shgo_fable_recommendations.md`). Python
cross-validation fixtures pin what actually matches SciPy: Sobol sequence
values (bit-exact), the initial simplicial triangulation (vertices and edges),
vertex caching and minimizer detection, and end-to-end optimum values for
single-iteration runs. Not identical to SciPy by design: the local optimizer
(BOBYQA instead of SLSQP), Sobol-sequence continuation across iterations
(SciPy's own is buggy), the Delaunay edge set (SciPy's `vf_to_vv` drops most
simplex edges for `dim >= 3`; the faithful reproduction is available as
`ConnectivityMethod::DelaunayScipyCompat`), the de-duplication of `xl`, and the
simplicial refinement order beyond the first iteration.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
shgo-rs = "0.1"
```

### Quick Start

```rust
use shgo::{Shgo, ShgoOptions};

// Minimize the Rosenbrock function
let result = Shgo::new(
    |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
    vec![(-5.0, 5.0), (-5.0, 5.0)],
)
.minimize()
.unwrap();

println!("Minimum: {:.6} at {:?}", result.fun, result.x);
// Minimum: 0.000000 at [1.0, 1.0]
```

### Builder Pattern with Options

```rust
use shgo::{Shgo, ShgoOptions, SamplingMethod};

let result = Shgo::new(
    |x: &[f64]| {
        // N-dimensional Ackley function
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let sum_cos: f64 = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    },
    vec![(-5.0, 5.0); 3],
)
.with_options(ShgoOptions {
    maxiter: Some(3),
    n: 128,
    sampling_method: SamplingMethod::Sobol,
    workers: None,  // use all CPU cores
    ..Default::default()
})
.minimize()
.unwrap();

assert!(result.fun < 1e-6);
```

### Parallel Local Minimization

SHGO-RS uses [rayon](https://docs.rs/rayon) to parallelize the local
minimization phase. Control parallelism via the `workers` option:

```rust
use shgo::{Shgo, ShgoOptions};

// Serial (single-threaded)
let result_serial = Shgo::new(objective, bounds.clone())
    .with_options(ShgoOptions {
        workers: Some(1),
        maxiter: Some(4),
        ..Default::default()
    })
    .minimize()
    .unwrap();

// Parallel (all available CPU cores)
let result_parallel = Shgo::new(objective, bounds)
    .with_options(ShgoOptions {
        workers: None,   // None = all cores via rayon
        maxiter: Some(4),
        ..Default::default()
    })
    .minimize()
    .unwrap();
```

Control rayon's thread pool size via the `RAYON_NUM_THREADS` environment
variable.

### Accessing All Local Minima

```rust
use shgo::{Shgo, ShgoOptions};

let result = Shgo::new(eggholder, vec![(-512.0, 512.0), (-512.0, 512.0)])
    .with_options(ShgoOptions {
        maxiter: Some(5),
        ..Default::default()
    })
    .minimize()
    .unwrap();

println!("Global minimum: {:.6} at {:?}", result.fun, result.x);
println!("All {} local minima found:", result.xl.len());
for (i, (x, f)) in result.xl.iter().zip(result.funl.iter()).enumerate() {
    println!("  [{}] f={:.6} at {:?}", i, f, x);
}
```

## Configuration Reference

### `ShgoOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `iters` | `Option<usize>` | `Some(1)` | Refinement iterations (passes) |
| `maxiter` | `Option<usize>` | `None` | Hard iteration limit |
| `maxfev` | `Option<usize>` | `None` | Max function evaluations |
| `maxev` | `Option<usize>` | `None` | Max sampling evaluations |
| `maxtime` | `Option<f64>` | `None` | Wall-clock time limit (seconds) |
| `f_min` | `Option<f64>` | `None` | Known global minimum value |
| `f_tol` | `f64` | `1e-4` | Tolerance for `f_min` stopping criterion |
| `n` | `usize` | `0` (auto) | Sampling points per iteration |
| `sampling_method` | `SamplingMethod` | `Simplicial` | Sampling strategy |
| `connectivity_method` | `ConnectivityMethod` | `Delaunay` | Sobol-mode sampling-graph construction |
| `knn_neighbors` | `Option<usize>` | `None` (auto: 2·dim+1) | Neighbor count for KNN/HNSW/ScaNN |
| `minimize_every_iter` | `bool` | `true` | Run local minimization each iter |
| `maxiter_local` | `Option<usize>` | `None` | Max local minimizations per iter |
| `disp` | `usize` | `0` | Verbosity (0=silent, 1=summary, 2=detailed) |
| `local_options` | `LocalOptimizerOptions` | BOBYQA, tol 1e-12 | Local solver algorithm + tolerances (`local_options.algorithm`). Note `LocalOptimizerOptions::default()` on its own uses `ftol_rel = 1e-8`. |
| `workers` | `Option<usize>` | `None` | Thread count (`None` = all cores) |
| `xl_dedup_rtol` | `f64` | `1e-4` | Two local results are the same minimum when every coordinate agrees to this fraction of the bounds width (also stops re-minimizing sampling points that sit on a known minimum); `0` = bitwise only |
| `xl_dedup_ftol` | `f64` | `1e-6` | Relative function-value agreement additionally required for merging |
| `knn_auto` | `Option<KnnAuto>` | `None` | Pick k from the candidate-count curve to fit a local-run budget (k-NN connectivity only) |
| `min_candidate_persistence` | `Option<f64>` | `None` | Drop candidates whose basin persistence is at or below this, before any local run |
| `max_candidates_by_persistence` | `Option<usize>` | `None` | Keep at most this many candidates, most persistent first |
| `explore_from_known_minima` | `bool` | `false` | Also re-run the local optimizer from already-found minima (see below) |
| `robustness_probe` | `Option<RobustnessProbe>` | `None` | After optimization, measure each retained minimum's sensitivity to parameter perturbations on a deterministic stencil |
| `robust_polish` | `Option<RobustPolish>` | `None` | Re-optimize the best minima on the stencil-smoothed objective; results in `robust_minima` |
| `f_min` + `f_tol` | — | — | Precision-based early stopping |

`iters: None` is only accepted together with another stopping criterion
(`maxiter`, `maxfev`, `maxev`, `maxtime` or `f_min`); otherwise `minimize()`
returns `ShgoError::InvalidOption` instead of looping forever.

### Choosing the k-NN neighbour count

In Sobol + k-NN mode the sampling graph is a k-nearest-neighbour graph, so `k`
trades superfluous local runs (small k, many spurious candidates) against
missed minima (large k). The number of candidates `|M_k|` is non-increasing in
k, and each candidate costs one local minimization, which makes k a dial on the
local-search budget. `knn_auto` turns that dial for you from a single neighbour
pass and **zero extra objective evaluations**:

```rust
use shgo::{KnnAuto, ShgoOptions, SamplingMethod, ConnectivityMethod};

let options = ShgoOptions {
    sampling_method: SamplingMethod::Sobol,
    connectivity_method: ConnectivityMethod::KNearestNeighbors,
    n: 16384,
    knn_auto: Some(KnnAuto::with_budget(60)),  // at most ~60 local runs/iteration
    ..Default::default()
};
// result.knn_selection holds the chosen k and the whole |M_k| curve.
```

If the curve cannot reach the budget within `k_max`, the largest k is used and
the pool is simply as small as that k allows. Without `knn_auto`, `k` comes
from `knn_neighbors` (default `2·dim + 1`, which is only safe up to roughly
n = 1024 — measured floors for a unimodal surface at n = 16384 are k ≈ 16, 24,
40 and 60 at dim 4, 6, 8 and 10).

### Pruning the candidate pool by basin persistence

`min_candidate_persistence` and `max_candidates_by_persistence` drop candidates
whose basin merges into a deeper one at a low saddle — sampling artefacts
rather than distinct basins — before any local run happens. Both cost `O(V·k)`
arithmetic and no objective evaluations, and neither ever prunes the
lowest-cost candidate.

Persistence describes the **sampled** landscape, so when sampling is coarse
relative to the number of minima the basin that polishes deepest need not be a
prominent one. On a 2^d-well test function with 16384 points in 8 dimensions it
ranked 231st of 246 basins by persistence. Treat these as breadth-of-map knobs,
keep any threshold near the objective's noise floor, and do not rely on them to
preserve the global optimum.

### Robustness to parameter perturbations

Two opt-in post-optimization steps for problems whose answer must survive
small parameter changes. Both use the same deterministic stencil around a
point — the `2·dim` axis steps at a relative radius plus a small Sobol cloud in
that box, clipped to the bounds, with constraint-violating points skipped —
and both add their evaluations to `nfev` while leaving `x`, `fun`, `xl` and
`funl` untouched.

```rust
use shgo::{RobustnessProbe, RobustPolish, ShgoOptions};

let options = ShgoOptions {
    // 3 % of each parameter's range, 8 Sobol points besides the axis steps
    robustness_probe: Some(RobustnessProbe::new(0.03, 8)),
    // re-optimize the 3 most robust minima on the stencil-averaged objective
    robust_polish: Some(RobustPolish::new(0.03, 8, 3)),
    ..Default::default()
};
// result.robustness[i]: f_center, robust_value (mean by default; Max and CVaR
//   available), f_mean, f_max, f_std, worst_axis, n_feasible ...
// result.robust_minima: x, robust_value, f_center per polished minimum
```

The probe is a **polished-point** measurement. The basin statistics
(`compute_basin_stats`) describe the sampled cloud's catchments instead, and at
realistic densities a basin's members are few and mostly far from its minimum:
on a four-basin test at 8 192 points in 4 dimensions every basin's `f_median`
was within 4e-4 of zero, while the probe ranked the basins by their actual
widths and named the fragile axis. Use the cloud statistics for "how much of
the space drains here", and the probe for "what happens if these parameters
wobble".

### `explore_from_known_minima`

Every minimum re-inserted into the next Sobol iteration is a graph minimizer,
so re-running the local optimizer from it is possible (and is what this crate
did before September 2026). With BOBYQA's large initial trust region such a run
occasionally escapes to a deeper neighbouring basin, which made it an
accidental restart heuristic. Measured, it is not worth the default: on a
well-separated multi-well function it found **exactly** the same minima for
29-67 % more local evaluations, and on Rastrigin it found more minima roughly
in proportion to the extra cost. The global optimum was found either way in
every case. The option exists to reproduce the old behaviour.

### Sampling Methods

| Variant | Description |
|---|---|
| `SamplingMethod::Simplicial` | Topology-aware simplicial complex (default). Auto-scales: `2^dim + 1` points. Best for low-to-mid dimensions. |
| `SamplingMethod::Sobol` | Quasi-random Sobol sequence (128 points default). Better coverage in higher dimensions. |

### Connectivity Methods (Sobol mode)

The sampling graph used for topological minimizer detection is configurable via
`connectivity_method`:

| Variant | Description |
|---|---|
| `ConnectivityMethod::Delaunay` | QHull Delaunay triangulation, full 1-skeleton (default; every simplex edge, as in the SHGO paper). Cost grows combinatorially with dimension — impractical above ~7 dims. |
| `ConnectivityMethod::DelaunayScipyCompat` | Delaunay with SciPy's `vf_to_vv` quirk reproduced (for `dim >= 3` only the first three vertices of each simplex are connected, which yields hundreds of spurious minimizer candidates). Parity testing only. |
| `ConnectivityMethod::KNearestNeighbors` | Exact brute-force k-NN (rayon-parallel, O(n²·d)). Recommended for dim ≳ 7. |
| `ConnectivityMethod::HNSW` | Approximate k-NN via `hnsw_rs` (the query point itself is excluded, so `k` means the same as for exact k-NN). Only pays off for very large point sets. |
| `ConnectivityMethod::ScaNN` | Quantized approximate k-NN via `vecstore` (experimental; falls back to exact k-NN on failure). |

All methods build the graph over the full cumulative point cloud each iteration,
matching SciPy's re-triangulation semantics.

### Local Optimizer Algorithms

| Variant | NLopt Algorithm | Notes |
|---|---|---|
| `LocalOptimizer::Bobyqa` | `LN_BOBYQA` | Default. Derivative-free, supports bounds. |
| `LocalOptimizer::Cobyla` | `LN_COBYLA` | Supports nonlinear inequality constraints. |
| `LocalOptimizer::Slsqp` | `LD_SLSQP` | Gradient-based, sequential least squares; supports inequality constraints. Gradients by forward finite differences, evaluated in parallel (rayon) and counted in `nfev`. |
| `LocalOptimizer::Lbfgs` | `LD_LBFGS` | Limited-memory BFGS, gradient-based (finite-difference gradients as above). |
| `LocalOptimizer::NelderMead` | `LN_NELDERMEAD` | Simplex method, no bounds. |
| `LocalOptimizer::Praxis` | `LN_PRAXIS` | Principal axis method. |
| `LocalOptimizer::NewuoaBound` | `LN_NEWUOA_BOUND` | NEWUOA with bound constraints. |
| `LocalOptimizer::Sbplx` | `LN_SBPLX` | Subplex: Nelder-Mead restarted on low-dimensional subspaces. Tolerates noise and non-smoothness, scales better than Nelder-Mead. |
| `LocalOptimizer::Cmaes` | — (`cmaes` crate) | CMA-ES, population based. Bounds enforced by per-coordinate normalisation plus reflection; constraints upgrade to COBYLA. Seeded from the start point, so runs are deterministic. Tuned via `local_options.cmaes` (`CmaesOptions`: `sigma0`, `population_size`, `seed`, `parallel_eval`, `eval_final_mean`). See [Choosing between BOBYQA, Subplex and CMA-ES](#choosing-between-bobyqa-subplex-and-cma-es). |

## Termination Criteria

| Condition | Description |
|---|---|
| `iters` exhausted | Default stopping criterion (1 pass) |
| `maxiter` reached | Hard iteration cap |
| `maxfev` reached | Function evaluation budget exceeded |
| `maxev` reached | Sampling evaluation budget exceeded |
| `maxtime` elapsed | Wall-clock time limit reached |
| `f_min` + `f_tol` | Precision convergence: `(f_best - f_min) / |f_min| ≤ f_tol` |

## Parallelization

Function evaluations during **local minimization** are parallelized using
[rayon](https://docs.rs/rayon). The parallelism model:

- Each candidate in the minimizer pool is dispatched to rayon's work-stealing
  thread pool independently.
- Serial execution (`workers: Some(1)`) reproduces the same minimizer
  sequence as single-threaded mode for reproducibility.
- Parallel mode (`workers: None`) is most beneficial when:
  - There are many local minima candidates per iteration.
  - The objective function is moderately to very expensive (> ~0.1 ms per
    evaluation).
  - Dimensionality is ≥ 3 (more candidates per iteration).

### When to Use Parallel Mode

| Objective cost | Recommendation |
|---|---|
| < 10 µs | `workers: Some(1)` — rayon overhead dominates |
| 0.1–10 ms | `workers: None` — scales well with core count |
| > 10 ms | `workers: None` — near-linear speedup |

### Choosing the local optimizer for parallelism

The candidate pool is parallelised across candidates, and each candidate's
local minimization is a serial chain. A gradient-based method (`Slsqp`,
`Lbfgs`) additionally fans its `dim` finite differences across threads, so
which algorithm is fastest depends on how the pool compares with the core
count. Measured on 16 cores:

| regime | fastest | note |
|---|---|---|
| 25 candidates, 15 dims, 1.4 ms objective | BOBYQA (4.8 s vs 8.2 s for `Lbfgs`) | the pool already saturates the threads; the gradient's inner parallelism only adds scheduling |
| 1 candidate, 12 dims, 0.4 ms objective | `Slsqp` (0.019 s vs 0.037 s) | half the wall clock on 18 % *more* evaluations, because the gradient is 12-wide |
| evaluations are the scarce resource | `Lbfgs` | 2.5x fewer objective evaluations than BOBYQA in the 15-dimensional case |

BOBYQA remains the default. Consider a gradient method when the candidate pool
is narrower than the core count, which is the usual state of later iterations
and of a small `maxiter_local`.

### Choosing between BOBYQA, Subplex and CMA-ES

`Sbplx` (Rowan's subplex, via NLopt) and `Cmaes` (the [`cmaes`](https://docs.rs/cmaes)
crate) are the derivative-free alternatives for basins that are not smooth at
the scale the local optimizer works on. `examples/local_optimizer_benchmark.rs`
compares them with BOBYQA, L-BFGS and Nelder-Mead on landscapes that each break
one assumption of the model-based methods:

| problem | landscape | what it stresses |
|---|---|---|
| `rosen-6d` | Rosenbrock | smooth curved valley (control) |
| `rotated-10d` | rotated ellipsoid, condition 1e6 | non-separable, ill-conditioned |
| `rugged-4d/10d` | bowl + ripples of period 1/3 and amplitude 0.3 | ~8^dim local minima below the sampling resolution |
| `noisy-6d` | ellipsoid + frozen white noise of amplitude 0.5 | noise 50x the objective at the target radius |
| `step-5d` | ellipsoid quantised to plateaus of width 0.1 | zero gradient everywhere |

Settings: `ftol_rel 1e-12`, `xtol_rel 1e-8`, `maxeval 1000·dim`, CMA-ES
`sigma0 0.25` of the bound width. `Cmaes-4xpop` sets `population_size` to four
times the crate default `4 + floor(3 ln dim)`, the standard remedy for
multimodal or noisy objectives. "reached" counts starts that ended within the
problem's tolerance of the optimum (1e-6 for the smooth problems, 1e-3 for the
rugged bowl, distance 0.1 for the noisy one, exactly the zero plateau for the
step function); the error is `f - f*`, or `|x - x*|` for the noisy problem.
The two Rosenbrock misses are its second local minimum (`f ≈ 3.98`).

**Local optimizer alone, 8 fixed starts per problem:**

| problem | optimizer | reached | median err | worst err | median nfev | time ms |
|---|---|---|---|---|---|---|
| rosen-6d | Bobyqa | 6/8 | 3.4e-14 | 4.0e0 | 485 | 11.4 |
| rosen-6d | Lbfgs | 7/8 | 5.7e-11 | 4.0e0 | 525 | 25.5 |
| rosen-6d | NelderMead | 6/8 | 7.0e-15 | 4.0e0 | 960 | 1.3 |
| rosen-6d | Sbplx | 0/8 | 5.3e-1 | 4.4e0 | 6000 | 3.7 |
| rosen-6d | Cmaes | 7/8 | 6.1e-14 | 4.0e0 | 2770 | 22.4 |
| rosen-6d | Cmaes-4xpop | 8/8 | 2.7e-14 | 2.6e-11 | 6014 | 38.5 |
| rotated-10d | Bobyqa | 8/8 | 2.4e-12 | 6.0e-11 | 2972 | 187.8 |
| rotated-10d | Lbfgs | 8/8 | 3.2e-7 | 3.2e-7 | 830 | 35.5 |
| rotated-10d | NelderMead | 8/8 | 1.7e-14 | 2.7e-14 | 4933 | 17.1 |
| rotated-10d | Sbplx | 0/8 | 5.7e1 | 1.8e2 | 10000 | 16.9 |
| rotated-10d | Cmaes | 8/8 | 1.6e-13 | 3.2e-13 | 4757 | 64.4 |
| rotated-10d | Cmaes-4xpop | 8/8 | 9.2e-14 | 7.4e-13 | 8162 | 87.4 |
| rugged-4d | Bobyqa | 4/8 | 5.4e-2 | 3.2e-1 | 80 | 1.2 |
| rugged-4d | Lbfgs | 0/8 | 1.1e0 | 2.5e0 | 120 | 8.8 |
| rugged-4d | NelderMead | 1/8 | 2.1e-1 | 4.3e-1 | 329 | 0.5 |
| rugged-4d | Sbplx | 3/8 | 1.1e-1 | 2.1e-1 | 372 | 0.4 |
| rugged-4d | Cmaes | 3/8 | 1.1e-1 | 2.1e-1 | 962 | 7.2 |
| rugged-4d | Cmaes-4xpop | 8/8 | 6.7e-14 | 1.9e-13 | 2290 | 15.1 |
| rugged-10d | Bobyqa | 0/8 | 2.7e-1 | 6.4e-1 | 204 | 11.0 |
| rugged-10d | Lbfgs | 0/8 | 1.8e0 | 3.8e0 | 363 | 14.8 |
| rugged-10d | NelderMead | 0/8 | 1.4e0 | 3.4e0 | 1408 | 4.9 |
| rugged-10d | Sbplx | 2/8 | 2.7e-1 | 6.4e-1 | 928 | 1.5 |
| rugged-10d | Cmaes | 1/8 | 2.1e-1 | 4.3e-1 | 2467 | 33.3 |
| rugged-10d | Cmaes-4xpop | 8/8 | 1.5e-13 | 3.2e-13 | 5762 | 60.2 |
| noisy-6d | Bobyqa | 2/8 | 1.9e-1 | 4.6e-1 | 105 | 2.2 |
| noisy-6d | Lbfgs | 0/8 | 5.5e0 | 7.1e0 | 84 | 6.2 |
| noisy-6d | NelderMead | 0/8 | 7.3e-1 | 4.0e0 | 398 | 0.6 |
| noisy-6d | Sbplx | 1/8 | 2.1e-1 | 3.5e-1 | 536 | 0.5 |
| noisy-6d | Cmaes | 3/8 | 1.3e-1 | 3.5e-1 | 6005 | 60.1 |
| noisy-6d | Cmaes-4xpop | 8/8 | 4.1e-2 | 7.3e-2 | 6014 | 53.3 |
| step-5d | Bobyqa | 3/8 | 1.0e-2 | 1.6e-1 | 63 | 1.3 |
| step-5d | Lbfgs | 0/8 | 1.1e3 | 1.7e3 | 6 | 1.0 |
| step-5d | NelderMead | 3/8 | 3.8e-1 | 6.1e0 | 158 | 0.3 |
| step-5d | Sbplx | 7/8 | 0.0e0 | 1.0e-1 | 158 | 0.2 |
| step-5d | Cmaes | 8/8 | 0.0e0 | 0.0e0 | 666 | 7.0 |
| step-5d | Cmaes-4xpop | 8/8 | 0.0e0 | 0.0e0 | 1282 | 10.2 |

**Through SHGO** (Sobol + k-NN, `maxiter: 2`, `n` = 128 to 512): the minimizer
pool hands every method several starts, one of them usually close to the
optimum, so on four of the six problems every optimizer except Subplex on the
two smooth ones reached the optimum. The two that still discriminate:

| problem | optimizer | reached | err | nfev | nlfev | minima |
|---|---|---|---|---|---|---|
| rugged-10d | Bobyqa | yes | 5.4e-14 | 1187 | 162 | 1 |
| rugged-10d | Lbfgs | no | 2.2e0 | 1234 | 209 | 1 |
| rugged-10d | NelderMead | no | 5.4e-1 | 2087 | 1062 | 1 |
| rugged-10d | Sbplx | no | 5.4e-1 | 1836 | 811 | 1 |
| rugged-10d | Cmaes | no | 3.2e-1 | 3467 | 2442 | 1 |
| rugged-10d | Cmaes-4xpop | yes | 3.5e-13 | 6587 | 5562 | 1 |
| noisy-6d | Bobyqa | yes | 9.6e-2 | 1520 | 1003 | 9 |
| noisy-6d | Lbfgs | no | 4.3e0 | 1668 | 1155 | 10 |
| noisy-6d | NelderMead | no | 1.8e-1 | 4394 | 3877 | 9 |
| noisy-6d | Sbplx | no | 1.6e-1 | 4326 | 3809 | 7 |
| noisy-6d | Cmaes | yes | 2.4e-2 | 60567 | 60050 | 10 |
| noisy-6d | Cmaes-4xpop | yes | 2.7e-2 | 60657 | 60140 | 10 |

(On `rugged-10d` the single candidate happened to sit in the central basin,
which is why BOBYQA reached it there but from none of the eight fixed starts;
on `noisy-6d` BOBYQA's 0.096 is just inside the 0.1 tolerance where CMA-ES
lands at 0.025.)

What to take from it:

- **Smooth basin: keep BOBYQA.** It is the most precise and, with L-BFGS, the
  cheapest. CMA-ES needs 2 to 6 times the evaluations for the same answer on the
  smooth problems and 12 times with the enlarged population; nothing is gained.
- **Subplex is a plateau specialist, not a general fallback.** On the quantised
  objective it is perfect from 7 of 8 starts at 158 evaluations, where BOBYQA's
  quadratic model sees a flat function (3 of 8) and CMA-ES needs 4 times the
  evaluations. But it searches coordinate subspaces, so it fails outright on the
  curved Rosenbrock valley and the rotated ill-conditioned ellipsoid, both alone
  and through SHGO.
- **CMA-ES only pays off with an enlarged population.** With the default
  population it is no better than BOBYQA on the rugged and noisy problems while
  costing 10 to 50 times the evaluations. With `population_size` at four times
  the default it is the only method that reaches the bottom of the rugged bowl
  from every start in 4 and 10 dimensions, and the only one within the target
  radius on the noisy ellipsoid from every start. That costs 20 to 30 times
  BOBYQA's evaluations on those problems; it is worth it only when the basins
  SHGO identifies are rugged or noisy below the sampling resolution, so BOBYQA
  keeps returning a ripple minimum.
- **Cap `maxeval` explicitly for CMA-ES.** The default of 1000 evaluations is
  too small for CMA-ES with an enlarged population beyond about five dimensions
  (use around `1000·dim`), and under noise the tolerance criteria never trigger,
  so every candidate runs to the cap (the 60 000 evaluations on `noisy-6d`).
- **`sigma0` is relative to the bounds the local run receives.** In simplicial
  mode those are the locally convex bounds, so 0.25 is a quarter of the cell.
  In Sobol mode they are the global bounds, so 0.25 is a global-scale search from
  each candidate; lower it (0.1 in the calibration runs was as good or better on
  the smooth problems) when the candidates are already good.
- **Runs are deterministic.** Each CMA-ES run is seeded from its starting point,
  so the parallel minimizer pool gives the same result as `workers: Some(1)`;
  change `CmaesOptions::seed` to get a different realisation.

```rust
use shgo::{CmaesOptions, LocalOptimizer, LocalOptimizerOptions, Shgo, ShgoOptions};

let options = ShgoOptions {
    local_options: LocalOptimizerOptions {
        algorithm: LocalOptimizer::Cmaes,
        maxeval: Some(10_000),
        cmaes: CmaesOptions {
            population_size: Some(40), // 4x the default for 10 dimensions
            ..Default::default()
        },
        ..Default::default()
    },
    ..Default::default()
};
```

## SciPy Correspondence

### Module / Function Mapping

| SciPy (Python) | Rust Equivalent | File |
|---|---|---|
| `scipy.optimize.shgo()` | `Shgo::minimize()` | `src/shgo.rs` |
| `SHGO.__init__()` | `Shgo::new()` + `Shgo::with_options()` | `src/shgo.rs` |
| `SHGO.iterate()` | `Shgo::iterate()` | `src/shgo.rs` |
| `SHGO.find_minima()` | `Shgo::find_minima()` | `src/shgo.rs` |
| `SHGO._sampling()` | sampling dispatch | `src/shgo.rs` |
| `SHGO._sampling_simplicial()` | `SamplingMethod::Simplicial` path | `src/shgo.rs` |
| `SHGO._sampling_sobol()` | `SamplingMethod::Sobol` path | `src/shgo.rs` |
| `SHGO._minimizers_pool()` | minimizer pool construction | `src/shgo.rs` |
| `Complex` | `Complex` | `src/complex.rs` |
| `Vertex` / `VertexCache` | `Vertex` / `VertexCache` | `src/vertex.rs` |
| `Coordinates` | `Coordinates` | `src/coordinates.rs` |
| `_Sobol` (Joe-Kuo) | `Sobol` | `src/sobol.rs` |

### `OptimizeResult` Fields

| Field | Type | Description |
|---|---|---|
| `x` | `Vec<f64>` | Best parameter vector found |
| `fun` | `f64` | Function value at `x` |
| `xl` | `Vec<Vec<f64>>` | Locations of all local minima discovered |
| `funl` | `Vec<f64>` | Function values at all local minima |
| `success` | `bool` | Whether optimization succeeded |
| `message` | `String` | Human-readable status message |
| `nfev` | `usize` | Total function evaluations |
| `nit` | `usize` | Total iterations |
| `nlfev` | `usize` | Total local minimization evaluations |

## C/C++ FFI

The crate exposes a C-compatible FFI for use from C and C++ programs.
Headers are in `include/shgo.h` (C) and `include/shgo.hpp` (C++).

```c
#include "shgo.h"

double objective(const double* x, size_t n, void* user_data) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += x[i] * x[i];
    return sum;
}

int main() {
    double lb[] = {-5.0, -5.0};
    double ub[] = {5.0,  5.0};
    ShgoOptions opts = shgo_default_options();
    opts.maxiter = 3;

    ShgoResult result;
    ShgoStatus status = shgo_minimize(objective, NULL, 2, lb, ub, &opts, &result);

    if (status == SHGO_SUCCESS) {
        printf("f = %f at [%f, %f]\n", result.fun, result.x[0], result.x[1]);
        shgo_result_free(&result);
    }
    return 0;
}
```

See `examples_cpp/` for full C and C++ examples.

## Performance

Rust serial performance is approximately 0.8–1.2× Python/SciPy for a
comparable local optimizer. For expensive objective functions, parallel mode
provides near-linear speedup scaling with available CPU cores.

Benchmark summary (Release build, `RUSTFLAGS="-C target-cpu=native"`):

```
cargo build --examples --release
./target/release/examples/parallel_benchmark
```

## Verification & Testing

This implementation was verified against SciPy's SHGO through a comprehensive
cross-validation suite:

- **Sobol sequences** — direction numbers and sequence values match SciPy
  `scipy.stats.qmc.Sobol` exactly.
- **Vertex operations** — midpoints, field evaluation, and minimizer detection
  match the Python `Vertex` / `VertexCache` classes.
- **Simplicial complex** — the initial triangulation (vertex count and 2-D
  connectivity) matches SciPy; per-iteration growth matches SciPy's `+n`
  semantics. Refinement order beyond the first iteration is deterministic but
  not identical to SciPy's.
- **Minimizer results** — end-to-end fixtures (`tests/generate_e2e_fixtures.py`)
  record `scipy.optimize.shgo` results for sphere/Rosenbrock/constrained cases
  across both sampling modes; `test_end_to_end_matches_scipy` replays them in
  Rust and asserts agreement of the optimum.
- **Regression tests** for the behaviours fixed in September 2026: deterministic
  refinement, no re-minimization of known minima, full-skeleton Delaunay
  connectivity, working finite-difference gradients for SLSQP/L-BFGS, and
  constraints honoured by the public local-minimization API.

Run the full test suite:

```bash
cargo test
```

Cross-validate against Python (requires SciPy):

```bash
python tests/generate_fixtures.py   # regenerate JSON fixtures
cargo test --test cross_validation  # run Rust ↔ Python comparison
```

## References

- Endres, S.C., Sandrock, C. & Focke, W.W. "A simplicial homology algorithm
  for Lipschitz optimisation." *J Global Optim* 72, 181–217 (2018).
- Endres, S.C. "SHGO: Simplicial Homology Global Optimisation."
  <https://stefan-endres.github.io/shgo/>
- SciPy SHGO source:
  <https://github.com/scipy/scipy/blob/main/scipy/optimize/_shgo.py>
- Joe, S. & Kuo, F.Y. "Constructing Sobol sequences with better two-dimensional
  projections." *SIAM J. Sci. Comput.* 30, 2635–2654 (2008).

## Attribution

The bulk of this implementation was generated using **Claude Sonnet 4.6**
(Anthropic) with an automated multi-step agentic workflow that iteratively
builds, tests, and refines code toward a specified goal. The SciPy SHGO Python
source code served as the sole reference material for the faithful Rust port.
Meticulous low-level cross-validation testing — covering Sobol sequences,
Delaunay triangulations, vertex caching, simplicial complex construction, and
final optimizer results — was used to ensure **100% fidelity** to the SciPy
implementation. Rayon-based parallelism for local minimization was added as an
extension beyond the original Python implementation.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE)
file for details.
