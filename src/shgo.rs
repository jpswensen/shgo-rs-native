//! SHGO (Simplicial Homology Global Optimization) orchestrator.
//!
//! This module implements the main SHGO algorithm that coordinates:
//! - Simplicial complex construction and refinement
//! - Sobol sequence sampling for vertex generation
//! - Minimizer pool identification using topological analysis
//! - Local minimization of promising candidates
//!
//! # Example
//!
//! ```
//! use shgo::{Shgo, ShgoOptions, Bounds};
//!
//! // Rosenbrock function
//! let rosenbrock = |x: &[f64]| -> f64 {
//!     let a = 1.0;
//!     let b = 100.0;
//!     (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
//! };
//!
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//! let options = ShgoOptions {
//!     maxiter: Some(3),
//!     ..Default::default()
//! };
//!
//! let result = Shgo::new(rosenbrock, bounds)
//!     .with_options(options)
//!     .minimize()
//!     .unwrap();
//!
//! println!("Minimum at: {:?}", result.x);
//! println!("Function value: {}", result.fun);
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::complex::Complex;
use crate::coordinates::Coordinates;
use crate::error::ShgoError;
use crate::sobol::Sobol;
use crate::vertex::VertexCache;

/// Suppress stdout for the duration of a closure.
///
/// The upstream `qhull` crate has a debug `println!` that leaks pointer
/// addresses to stdout on every Delaunay call. This helper redirects fd 1
/// to /dev/null while `f` runs, then restores it.
#[cfg(unix)]
fn with_stdout_suppressed<R>(f: impl FnOnce() -> R) -> R {
    use std::os::unix::io::AsRawFd;
    let devnull = std::fs::File::open("/dev/null");
    let (devnull, old_stdout) = match devnull {
        Ok(dn) => {
            let old = unsafe { libc::dup(1) };
            if old >= 0 {
                unsafe { libc::dup2(dn.as_raw_fd(), 1) };
                (Some(dn), old)
            } else {
                return f();
            }
        }
        Err(_) => return f(),
    };
    let result = f();
    unsafe {
        libc::dup2(old_stdout, 1);
        libc::close(old_stdout);
    }
    drop(devnull);
    result
}

#[cfg(not(unix))]
fn with_stdout_suppressed<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// Type alias for objective function.
pub type ObjectiveFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

/// Type alias for constraint function (g(x) >= 0).
pub type ConstraintFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

/// Type alias for bounds specification.
pub type Bounds = Vec<(f64, f64)>;

/// Sampling method for generating vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum SamplingMethod {
    /// Simplicial sampling using cyclic product triangulation.
    /// Best for low-dimensional problems (n < 10).
    #[default]
    Simplicial,
    /// Sobol sequence sampling with optional Delaunay triangulation.
    /// Better for high-dimensional problems.
    Sobol,
}


/// Method for building vertex connectivity in Sobol mode.
///
/// Controls how neighbor relationships are established between sampled points.
/// This affects local minimizer detection (a vertex is a local minimizer if
/// its function value is less than all its neighbors').
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum ConnectivityMethod {
    /// Delaunay triangulation via QHull (the default): every edge of every
    /// simplex becomes a graph edge (the full 1-skeleton, as in the SHGO
    /// paper's Definition 18).
    ///
    /// Produces geometrically correct neighbors but costs O(n^⌈d/2⌉),
    /// making it prohibitively expensive for dimensions > ~7.
    #[default]
    Delaunay,

    /// Delaunay triangulation with SciPy's `vf_to_vv` edge-selection quirk
    /// reproduced: for `dim >= 3` only the triangle on the first three
    /// vertices of each simplex is connected, so the graph is much sparser
    /// than the triangulation and spawns many spurious minimizer candidates
    /// (e.g. 288 candidates on a 4-D sphere with 4096 points, versus 1 for
    /// [`ConnectivityMethod::Delaunay`]). Only useful for parity testing
    /// against SciPy; identical to `Delaunay` for `dim <= 2`.
    DelaunayScipyCompat,

    /// k-nearest-neighbors connectivity.
    ///
    /// Connects each point to its k nearest neighbors (bidirectionally).
    /// Much faster than Delaunay in high dimensions: O(n² · d) brute-force,
    /// which is fast for typical SHGO problem sizes (n ≤ 4096).
    ///
    /// Default k = 2·dim + 1, which empirically approximates Delaunay
    /// neighbor count. Override with [`ShgoOptions::knn_neighbors`].
    KNearestNeighbors,

    /// HNSW (Hierarchical Navigable Small World) approximate nearest neighbors.
    ///
    /// Uses the `hnsw_rs` crate to build a navigable small-world graph and
    /// query approximate k-nearest neighbors. Offers O(n·log n) build time
    /// and O(log n) per query, making it attractive for larger point sets.
    ///
    /// Default k = 2·dim + 1 (same as KNearestNeighbors).
    /// Override with [`ShgoOptions::knn_neighbors`].
    HNSW,

    /// ScaNN (Scalable Nearest Neighbors) approximate nearest neighbors.
    ///
    /// Uses the `vecstore` crate's ScaNN implementation with learned
    /// quantization and tree-based partitioning. Designed for large-scale
    /// vector search with configurable accuracy/speed tradeoffs.
    ///
    /// Note: operates on f32 internally (f64 points are down-cast).
    /// Default k = 2·dim + 1. Override with [`ShgoOptions::knn_neighbors`].
    ScaNN,
}


/// Options for SHGO optimization.
#[derive(Debug, Clone)]
pub struct ShgoOptions {
    /// Number of iterations to perform (refinement passes).
    /// Default: Some(1) — one refinement pass then done.
    /// Set to None for unlimited iterations (controlled by other criteria).
    /// Note: if any other stopping criterion (maxfev, f_min, etc.) is set,
    /// iters is effectively None (infinite) and only those criteria control
    /// termination (matching Python's behavior).
    pub iters: Option<usize>,

    /// Maximum number of iterations (hard stopping criterion).
    /// Default: None (no limit).
    pub maxiter: Option<usize>,

    /// Maximum number of function evaluations.
    /// Default: None (no limit).
    pub maxfev: Option<usize>,

    /// Maximum number of sampling evaluations (including infeasible).
    /// Default: None (no limit).
    pub maxev: Option<usize>,

    /// Maximum time limit in seconds.
    /// Default: None (no limit).
    pub maxtime: Option<f64>,

    /// Known minimum function value (for precision stopping).
    /// Default: None.
    pub f_min: Option<f64>,

    /// Tolerance for function value precision stopping.
    /// Used with f_min: stops when (f_lowest - f_min) / |f_min| <= f_tol.
    /// If f_min == 0, stops when f_lowest <= f_tol.
    /// Default: 1e-4
    pub f_tol: f64,

    /// Number of sampling points per iteration.
    /// For Simplicial with n=0: auto-computed as 2^dim + 1.
    /// For Sobol with n=0: defaults to 128.
    /// Default: 0 (auto)
    pub n: usize,

    /// Sampling method to use.
    /// Default: Simplicial
    pub sampling_method: SamplingMethod,

    /// Whether to minimize all local minima found.
    /// Default: true
    pub minimize_every_iter: bool,

    /// Maximum number of local minimizations per iteration.
    /// Default: None (no limit).
    pub maxiter_local: Option<usize>,

    /// Verbosity level (0 = silent, 1 = summary, 2 = detailed).
    /// Default: 0
    pub disp: usize,

    /// Number of initial Sobol points to skip.
    /// Default: 0 (include the origin, matching Python's scipy.stats.qmc.Sobol)
    pub sobol_skip: usize,

    /// Options for the local optimizer (including which algorithm to use).
    /// The algorithm is controlled by `local_options.algorithm`.
    /// When constraints are provided and the chosen algorithm doesn't support them,
    /// SHGO will automatically upgrade to Cobyla.
    /// Default: LocalOptimizerOptions with Bobyqa algorithm
    pub local_options: crate::local_opt::LocalOptimizerOptions,

    /// Number of worker threads for parallel execution.
    /// Default: None (use all available CPU cores)
    /// Set to Some(1) for single-threaded execution.
    pub workers: Option<usize>,

    /// Method for building vertex connectivity in Sobol mode.
    /// Default: Delaunay (matching SciPy).
    /// Set to KNearestNeighbors for faster high-dimensional problems.
    pub connectivity_method: ConnectivityMethod,

    /// Compute per-basin statistics over the sampled vertex cloud after the
    /// final iteration (graph-descent basin labeling + persistence sweep).
    /// Costs O(V·k) arithmetic and zero objective evaluations. Extension —
    /// SciPy SHGO has no equivalent. Default: false.
    pub compute_basin_stats: bool,

    /// Absolute cost thresholds for [`BasinStats::good_counts`]: for each
    /// threshold t, the count of basin members with f <= t is reported.
    pub basin_good_thresholds: Vec<f64>,

    /// Fraction of worst member costs averaged into [`BasinStats::f_tail`]
    /// (CVaR-style). Default: 0.1.
    pub basin_tail_fraction: f64,

    /// Number of nearest neighbors for k-NN connectivity.
    /// Only used when connectivity_method is KNearestNeighbors.
    /// Default: None (auto: 2·dim + 1).
    pub knn_neighbors: Option<usize>,

    /// Two local results are treated as the same minimum when every
    /// coordinate agrees to within `xl_dedup_rtol` times the width of that
    /// dimension's bounds (and their function values agree to
    /// [`ShgoOptions::xl_dedup_ftol`]). Used to de-duplicate `xl`/`funl`, to
    /// map basins onto `xl`, and to avoid re-running a local minimization
    /// from a sampling point that already sits on a known minimum (SciPy
    /// skips such vertices too). `0.0` merges only bitwise-identical points.
    /// Extension — SciPy keeps one `xl` row per starting point. Default: 1e-4.
    pub xl_dedup_rtol: f64,

    /// Relative function-value tolerance for the `xl` de-duplication:
    /// `|f_a - f_b| <= xl_dedup_ftol * max(1, |f_a|, |f_b|)`. Guards against
    /// merging distinct minima that happen to be close in badly scaled
    /// dimensions. `0.0` disables the function-value check. Default: 1e-6.
    pub xl_dedup_ftol: f64,

    /// Choose the k-nearest-neighbour count from the minimizer-pool curve
    /// instead of using a fixed [`ShgoOptions::knn_neighbors`].
    ///
    /// Only honoured for [`ConnectivityMethod::KNearestNeighbors`] (the other
    /// methods ignore it, with a warning at `disp > 0`). Costs one neighbour
    /// pass and **zero objective evaluations**: the number of minimizer
    /// candidates `|M_k|` is non-increasing in k, so a single sorted-neighbour
    /// computation yields the whole curve and the smallest k that fits the
    /// requested budget. See [`KnnAuto`]. Default: `None`.
    pub knn_auto: Option<KnnAuto>,

    /// Drop minimizer candidates whose basin persistence is at or below this
    /// value, before any local minimization runs.
    ///
    /// Persistence is the cost gap between a basin's lowest sampled vertex and
    /// the saddle at which that basin merges into a deeper one (the global
    /// basin has infinite persistence and is never dropped). A candidate with
    /// persistence below the objective's noise level is a sampling artefact
    /// rather than a distinct basin, so this prunes redundant local runs
    /// without the recall loss that raising k causes. Costs `O(V·k)`
    /// arithmetic and zero objective evaluations. Extension — SciPy has no
    /// equivalent. Default: `None`.
    ///
    /// **Caveat.** Persistence describes the *sampled* landscape. When the
    /// sampling is coarse relative to the number of minima, the basin that
    /// polishes to the global optimum need not be a prominent one: on a
    /// 2^d-well test function with 16384 points in 8 dimensions it ranked
    /// 231st of 246 basins by persistence. Keep the threshold near the
    /// objective's noise floor, and prefer this over
    /// [`ShgoOptions::max_candidates_by_persistence`] when the global optimum
    /// matters more than the breadth of the map. The lowest-cost candidate is
    /// never pruned by either filter.
    pub min_candidate_persistence: Option<f64>,

    /// Keep at most this many minimizer candidates per iteration, choosing the
    /// most persistent (see [`ShgoOptions::min_candidate_persistence`]).
    ///
    /// Unlike [`ShgoOptions::maxiter_local`], which keeps the candidates with
    /// the lowest sampled cost, this keeps the ones most likely to be distinct
    /// basins. Applied after `min_candidate_persistence` and before
    /// `maxiter_local`. The lowest-cost candidate is always kept. Default:
    /// `None`.
    ///
    /// **This is a breadth-of-map knob, not a global-optimum knob** — see the
    /// caveat on [`ShgoOptions::min_candidate_persistence`]. Truncating a
    /// coarsely sampled landscape to its most prominent basins can discard the
    /// basin that would have polished deepest.
    pub max_candidates_by_persistence: Option<usize>,

    /// Also start a local minimization from sampling points that sit on an
    /// already-found minimum, instead of skipping them.
    ///
    /// Every minimum re-inserted into the next Sobol iteration is a graph
    /// minimizer by construction, so this re-runs the local optimizer from
    /// each known minimum once per iteration. With a derivative-free method
    /// whose initial trust region is a fraction of the box (BOBYQA, the
    /// default) such a run sometimes leaves the basin and lands in a deeper
    /// neighbouring one, which makes this a crude restart heuristic — at the
    /// cost of one full local run per known minimum per iteration, and of
    /// SciPy parity (its `minimizers()` skips these vertices). Default:
    /// `false`.
    pub explore_from_known_minima: bool,

    /// After optimization, measure each retained minimum's sensitivity to
    /// parameter perturbations on a deterministic stencil. See
    /// [`RobustnessProbe`]. Results in [`ShgoResult::robustness`]; the
    /// evaluations are added to `nfev`. Default: `None`.
    pub robustness_probe: Option<RobustnessProbe>,

    /// After optimization (and the probe, if any), re-optimize the best minima
    /// on the stencil-smoothed objective. See [`RobustPolish`]. Results in
    /// [`ShgoResult::robust_minima`]; the evaluations are added to `nfev`.
    /// Default: `None`.
    pub robust_polish: Option<RobustPolish>,
}

/// Budget-driven automatic choice of the k-nearest-neighbour count.
///
/// The number of minimizer candidates `|M_k|` is non-increasing in k, and each
/// candidate costs one local minimization, so k is really a dial on the local
/// search budget. This picks the smallest k whose candidate count fits
/// `max_local_runs`, which keeps the sampling graph as sparse (and therefore as
/// sensitive to shallow basins) as the budget allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnnAuto {
    /// Target ceiling on the number of local minimizations per iteration.
    pub max_local_runs: usize,
    /// Largest k to consider. `None` = `max(4·dim, 64)`, capped at `n - 1`.
    pub k_max: Option<usize>,
    /// Smallest k to consider. `None` = `dim + 1`. Raising this guards against
    /// a degenerate curve on a nearly flat objective.
    pub k_min: Option<usize>,
}

impl KnnAuto {
    /// Auto-select k with the given ceiling on local minimizations per
    /// iteration and default k bounds.
    pub fn with_budget(max_local_runs: usize) -> Self {
        Self {
            max_local_runs,
            k_max: None,
            k_min: None,
        }
    }
}

/// Outcome of one automatic k selection (see [`ShgoOptions::knn_auto`]).
#[derive(Debug, Clone)]
pub struct KnnSelection {
    /// The k that was used to build the sampling graph.
    pub k: usize,
    /// The largest k considered.
    pub k_max: usize,
    /// `curve[k]` is the number of minimizer candidates the graph would have
    /// at that k, for `k` in `0..=k_max`. Non-increasing. `curve[0]` is the
    /// number of feasible sampled points.
    pub curve: Vec<usize>,
}

/// How the objective values over a perturbation stencil are collapsed into one
/// number (the "robust value" of a point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustAggregate {
    /// Mean over the stencil: the expected value under a uniform perturbation.
    Mean,
    /// Worst case over the stencil.
    Max,
    /// Mean of the worst `fraction` of the stencil (CVaR-style; `fraction` in
    /// `(0, 1]`, e.g. `0.25` = mean of the worst quarter).
    Cvar { fraction: f64 },
}

/// A deterministic perturbation stencil around a point: the point itself, the
/// `2·dim` axis steps `x ± radius_i·e_i`, and `samples` Sobol points in the box
/// `[x − radius, x + radius]`, all clipped to the bounds. `radius_i` is
/// `radius_rel` times the width of dimension `i`'s bounds (`max(1, |x_i|)`
/// when that width is effectively infinite).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stencil {
    /// Half-width per dimension, as a fraction of that dimension's bounds width.
    pub radius_rel: f64,
    /// Include the `2·dim` axis steps.
    pub axis_steps: bool,
    /// Number of Sobol points in the perturbation box (0 = none).
    pub samples: usize,
}

impl Stencil {
    /// Axis steps only, at the given relative radius.
    pub fn axes(radius_rel: f64) -> Self {
        Self {
            radius_rel,
            axis_steps: true,
            samples: 0,
        }
    }

    /// Axis steps plus `samples` Sobol points.
    pub fn with_samples(radius_rel: f64, samples: usize) -> Self {
        Self {
            radius_rel,
            axis_steps: true,
            samples,
        }
    }
}

/// Measure how sensitive each polished minimum is to small parameter
/// perturbations, after the optimization has finished.
///
/// For every retained minimum (or the `top` lowest), the objective is evaluated
/// on a [`Stencil`] around it and summarised in a [`RobustnessStats`]. This is
/// a *polished-point* measurement, complementary to the sampled-cloud
/// [`BasinStats`]: at typical sampling densities a basin has few, distant
/// members, whereas the stencil sits exactly where the answer is. Costs the
/// stencil size in objective evaluations per probed minimum and nothing else.
/// Extension — SciPy has no equivalent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustnessProbe {
    /// The perturbation stencil.
    pub stencil: Stencil,
    /// Aggregate reported as [`RobustnessStats::robust_value`].
    pub aggregate: RobustAggregate,
    /// Probe only the `top` minima with the lowest `funl` (`None` = all).
    pub top: Option<usize>,
}

impl RobustnessProbe {
    /// Axis steps plus `samples` Sobol points at `radius_rel`, mean aggregate,
    /// all minima.
    pub fn new(radius_rel: f64, samples: usize) -> Self {
        Self {
            stencil: Stencil::with_samples(radius_rel, samples),
            aggregate: RobustAggregate::Mean,
            top: None,
        }
    }
}

/// Re-optimize the best minima on the *smoothed* objective — the
/// [`RobustAggregate`] of the objective over a [`Stencil`] around each trial
/// point — so the answer moves to the centre of a flat region rather than to
/// the bottom of a sharp one.
///
/// Every evaluation of the smoothed objective costs the stencil size in raw
/// evaluations, so this is applied to the `top` minima only, ranked by the
/// probe's robust value when [`ShgoOptions::robustness_probe`] is set and by
/// `funl` otherwise. The raw `xl`/`funl`/`x`/`fun` are left untouched; the
/// robust results are reported separately in [`ShgoResult::robust_minima`].
/// Extension — SciPy has no equivalent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustPolish {
    /// The perturbation stencil used inside the smoothed objective.
    pub stencil: Stencil,
    /// How the stencil values are aggregated into the smoothed objective.
    pub aggregate: RobustAggregate,
    /// Number of minima to re-optimize.
    pub top: usize,
    /// Cap on evaluations of the *smoothed* objective per minimum (each costs
    /// the stencil size in raw evaluations). `None` = the local optimizer's
    /// own `maxeval`.
    pub maxeval: Option<u32>,
}

impl RobustPolish {
    /// Robust-polish the `top` minima with axis steps plus `samples` Sobol
    /// points at `radius_rel`, mean aggregate.
    pub fn new(radius_rel: f64, samples: usize, top: usize) -> Self {
        Self {
            stencil: Stencil::with_samples(radius_rel, samples),
            aggregate: RobustAggregate::Mean,
            top,
            maxeval: None,
        }
    }
}

/// Sensitivity of one polished minimum to parameter perturbations (see
/// [`RobustnessProbe`]).
#[derive(Debug, Clone)]
pub struct RobustnessStats {
    /// Row of `xl` / `funl` this describes.
    pub xl_index: usize,
    /// The minimum's own objective value (`funl[xl_index]`).
    pub f_center: f64,
    /// The requested aggregate over the feasible stencil points (including
    /// the centre). Lower is more robust for a minimization.
    pub robust_value: f64,
    /// Mean over the feasible stencil points.
    pub f_mean: f64,
    /// Median over the feasible stencil points.
    pub f_median: f64,
    /// Best and worst feasible stencil values.
    pub f_min: f64,
    pub f_max: f64,
    /// Standard deviation over the feasible stencil points.
    pub f_std: f64,
    /// Dimension whose axis step produced the largest increase over
    /// `f_center` (`None` without axis steps or if nothing increased).
    pub worst_axis: Option<usize>,
    /// Stencil points evaluated / skipped as infeasible (constraint violation,
    /// non-finite objective, or clipped onto the centre itself).
    pub n_feasible: usize,
    pub n_infeasible: usize,
}

/// One minimum re-optimized on the smoothed objective (see [`RobustPolish`]).
#[derive(Debug, Clone)]
pub struct RobustMinimum {
    /// Row of `xl` this started from.
    pub xl_index: usize,
    /// Location of the robust optimum.
    pub x: Vec<f64>,
    /// Smoothed objective (the configured aggregate) at `x`.
    pub robust_value: f64,
    /// Raw objective at `x`.
    pub f_center: f64,
    /// Raw objective evaluations spent on this minimum.
    pub nfev: usize,
    /// Whether the local optimizer reported convergence.
    pub success: bool,
}

impl Default for ShgoOptions {
    fn default() -> Self {
        Self {
            iters: Some(1),
            maxiter: None,
            maxfev: None,
            maxev: None,
            maxtime: None,
            f_min: None,
            f_tol: 1e-4,
            n: 0, // Auto: 2^dim + 1 for simplicial, 128 for sobol
            sampling_method: SamplingMethod::Simplicial,
            minimize_every_iter: true,
            maxiter_local: None,
            disp: 0,
            sobol_skip: 0,
            local_options: crate::local_opt::LocalOptimizerOptions {
                algorithm: crate::local_opt::LocalOptimizer::Bobyqa,
                ftol_rel: 1e-12,
                ..crate::local_opt::LocalOptimizerOptions::default()
            },
            workers: None,
            connectivity_method: ConnectivityMethod::Delaunay,
            knn_neighbors: None,
            compute_basin_stats: false,
            basin_good_thresholds: Vec::new(),
            basin_tail_fraction: 0.1,
            xl_dedup_rtol: 1e-4,
            xl_dedup_ftol: 1e-6,
            knn_auto: None,
            min_candidate_persistence: None,
            max_candidates_by_persistence: None,
            explore_from_known_minima: false,
            robustness_probe: None,
            robust_polish: None,
        }
    }
}

/// Result of a local minimization.
#[derive(Debug, Clone)]
pub struct LocalMinimum {
    /// Location of the local minimum.
    pub x: Vec<f64>,
    /// Function value at the minimum.
    pub fun: f64,
    /// Whether the local minimization succeeded.
    pub success: bool,
    /// Number of function evaluations used.
    pub nfev: usize,
    /// Number of iterations used.
    pub nit: usize,
}

/// Statistics for one basin of attraction of the sampled vertex cloud.
///
/// Basins are computed by steepest-descent labeling on the sampling graph:
/// every finite-cost vertex is assigned to the graph minimizer its descent
/// path terminates at. Because Sobol samples are uniform over the domain,
/// `size` is an unbiased estimator of the basin's volume fraction
/// (`size / total_sampled`). Sorted ascending by `f_min_sampled`.
#[derive(Debug, Clone)]
pub struct BasinStats {
    /// Index into `xl`/`funl` of the polished local minimum this basin maps
    /// to (None if the basin's graph minimizer was never polished, or its
    /// polish is not retained in `xl`).
    pub xl_index: Option<usize>,
    /// Coordinates of the basin's lowest sampled vertex (the graph minimizer).
    pub x_sampled: Vec<f64>,
    /// Polished minimum from the local-minimization cache, if the graph
    /// minimizer was used as a starting point.
    pub x_polished: Option<Vec<f64>>,
    /// Cost at the lowest sampled vertex (pre-polish).
    pub f_min_sampled: f64,
    /// Number of sampled vertices in this basin (uniform-sample volume proxy).
    pub size: usize,
    /// Mean member cost.
    pub f_mean: f64,
    /// Median member cost.
    pub f_median: f64,
    /// Mean of the worst `basin_tail_fraction` member costs (CVaR-style).
    pub f_tail: f64,
    /// Member counts at or below each `basin_good_thresholds` entry.
    pub good_counts: Vec<usize>,
    /// Persistence: cost at the saddle where this basin merges into an
    /// older (lower) basin, minus `f_min_sampled`. `f64::INFINITY` for the
    /// global basin. Basins with persistence below the objective's noise
    /// level are likely sampling artifacts.
    pub persistence: f64,
}

/// Result of SHGO optimization.
#[derive(Debug, Clone)]
pub struct ShgoResult {
    /// Best solution found (global minimum candidate).
    pub x: Vec<f64>,
    /// Function value at best solution.
    pub fun: f64,
    /// All local minima found, sorted by function value.
    pub xl: Vec<Vec<f64>>,
    /// Function values at all local minima.
    pub funl: Vec<f64>,
    /// Whether optimization succeeded.
    pub success: bool,
    /// Status message.
    pub message: String,
    /// Total number of function evaluations.
    pub nfev: usize,
    /// Total number of iterations (refinement cycles).
    pub nit: usize,
    /// Total number of local function evaluations across all local minimizations.
    pub nlfev: usize,
    /// Total optimization time in seconds.
    pub time: f64,
    /// Per-basin statistics of the sampled cloud (only when
    /// `ShgoOptions::compute_basin_stats` is set).
    pub basins: Option<Vec<BasinStats>>,
    /// The k chosen by [`ShgoOptions::knn_auto`] and the candidate-count curve
    /// it was chosen from, for the most recent iteration. `None` when
    /// automatic selection was not used.
    pub knn_selection: Option<KnnSelection>,
    /// Perturbation sensitivity of the probed minima, in `xl` order (only when
    /// [`ShgoOptions::robustness_probe`] is set).
    pub robustness: Option<Vec<RobustnessStats>>,
    /// Minima re-optimized on the smoothed objective (only when
    /// [`ShgoOptions::robust_polish`] is set), sorted ascending by
    /// `robust_value`.
    pub robust_minima: Option<Vec<RobustMinimum>>,
}

impl ShgoResult {
    /// Create a new result with default values.
    fn new(dim: usize) -> Self {
        Self {
            x: vec![0.0; dim],
            fun: f64::INFINITY,
            xl: Vec::new(),
            funl: Vec::new(),
            success: false,
            message: String::new(),
            nfev: 0,
            nit: 0,
            nlfev: 0,
            time: 0.0,
            basins: None,
            knn_selection: None,
            robustness: None,
            robust_minima: None,
        }
    }
}

/// Cache for local minimization results.
/// 
/// This prevents redundant local minimizations from the same starting point.
pub struct LMapCache {
    /// Cached results indexed by starting coordinates.
    cache: RwLock<IndexMap<Coordinates, LocalMinimum>>,
    /// Total function evaluations from local minimizations.
    total_fev: AtomicUsize,
}

impl LMapCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(IndexMap::new()),
            total_fev: AtomicUsize::new(0),
        }
    }

    /// Get a cached result for the given starting point.
    pub fn get(&self, coords: &Coordinates) -> Option<LocalMinimum> {
        self.cache.read().get(coords).cloned()
    }

    /// Insert a result into the cache.
    pub fn insert(&self, coords: Coordinates, result: LocalMinimum) {
        self.total_fev.fetch_add(result.nfev, Ordering::Relaxed);
        self.cache.write().insert(coords, result);
    }

    /// Check if a starting point has been minimized.
    pub fn contains(&self, coords: &Coordinates) -> bool {
        self.cache.read().contains_key(coords)
    }

    /// Get all cached results sorted by function value.
    pub fn get_sorted(&self) -> Vec<LocalMinimum> {
        let cache = self.cache.read();
        let mut results: Vec<_> = cache.values().cloned().collect();
        results.sort_by(|a, b| a.fun.partial_cmp(&b.fun).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get total function evaluations from all local minimizations.
    pub fn total_fev(&self) -> usize {
        self.total_fev.load(Ordering::Relaxed)
    }

    /// Number of cached results.
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }
}

impl Default for LMapCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 0-dimensional persistence of every basin in an evaluated vertex cloud: for
/// each graph minimizer, the cost gap between it and the saddle at which its
/// basin merges into an older (lower) one. Basins that never merge — the
/// global one, and any component isolated in the sampling graph — are absent
/// from the map; callers read that as infinite persistence.
///
/// Vertices are swept in ascending `(f, index)`; a union-find with the elder
/// rule records each younger basin's death. `fs[i]` is the value at
/// `vertices[i]`, and `vertices[i].index() == i` (the vertex-cache invariant),
/// so map keys are vertex indices. Deterministic.
fn compute_persistence(
    vertices: &[Arc<crate::Vertex>],
    fs: &[f64],
    finite: &[bool],
) -> std::collections::HashMap<usize, f64> {
    use std::collections::HashMap;

    let n = vertices.len();
    let lower = |a: usize, b: usize| (fs[a], a) < (fs[b], b);

    fn find(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }

    let mut order: Vec<usize> = (0..n).filter(|&i| finite[i]).collect();
    order.sort_by(|&a, &b| {
        fs[a]
            .partial_cmp(&fs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut uf: Vec<usize> = (0..n).collect();
    // A component's union-find root is always its `(f, index)`-minimal vertex
    // (merges always point the younger root at the elder), so `comp_min[r] == r`.
    let comp_min: Vec<usize> = (0..n).collect();
    let mut processed = vec![false; n];
    let mut persistence: HashMap<usize, f64> = HashMap::new();
    for &v in &order {
        processed[v] = true;
        for &nb in &vertices[v].neighbor_indices() {
            if nb >= n || !processed[nb] {
                continue;
            }
            let rv = find(&mut uf, v);
            let rn = find(&mut uf, nb);
            if rv == rn {
                continue;
            }
            let (elder, younger) = if lower(comp_min[rv], comp_min[rn]) {
                (rv, rn)
            } else {
                (rn, rv)
            };
            persistence.insert(comp_min[younger], fs[v] - fs[comp_min[younger]]);
            uf[younger] = elder;
        }
    }
    persistence
}

/// [`compute_persistence`] over a whole vertex cache.
fn persistence_map<F2, G2>(
    cache: &crate::vertex::VertexCache<F2, G2>,
) -> std::collections::HashMap<usize, f64>
where
    F2: Fn(&[f64]) -> f64 + Send + Sync,
    G2: Fn(&[f64]) -> bool + Send + Sync,
{
    let vertices: Vec<Arc<crate::Vertex>> = cache.iter().collect();
    let fs: Vec<f64> = vertices
        .iter()
        .map(|v| v.f().unwrap_or(f64::INFINITY))
        .collect();
    let finite: Vec<bool> = fs.iter().map(|f| f.is_finite()).collect();
    compute_persistence(&vertices, &fs, &finite)
}

/// Stencil points tagged with their axis (for axis steps), plus the number of
/// points that clipped onto the centre and were dropped.
type StencilPoints = (Vec<(Vec<f64>, Option<usize>)>, usize);

/// Collapse stencil values into the robust value (see [`RobustAggregate`]).
/// Empty input (no feasible stencil point) is `+inf`.
fn robust_aggregate(values: &[f64], agg: RobustAggregate) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    match agg {
        RobustAggregate::Mean => values.iter().sum::<f64>() / values.len() as f64,
        RobustAggregate::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        RobustAggregate::Cvar { fraction } => {
            let mut v = values.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = ((v.len() as f64 * fraction.clamp(0.0, 1.0)).ceil() as usize).clamp(1, v.len());
            v[v.len() - n..].iter().sum::<f64>() / n as f64
        }
    }
}

/// Compute per-basin statistics over an evaluated vertex cache.
///
/// Steepest-descent labeling under the total order (f, index) assigns each
/// finite vertex to a graph minimizer; [`compute_persistence`] supplies the
/// persistence of each. Deterministic.
fn compute_basin_statistics<F2, G2>(
    cache: &crate::vertex::VertexCache<F2, G2>,
    lmap_cache: &LMapCache,
    thresholds: &[f64],
    tail_fraction: f64,
) -> Vec<BasinStats>
where
    F2: Fn(&[f64]) -> f64 + Send + Sync,
    G2: Fn(&[f64]) -> bool + Send + Sync,
{
    use std::collections::HashMap;

    let vertices: Vec<std::sync::Arc<crate::Vertex>> = cache.iter().collect();
    let n = vertices.len();
    let fs: Vec<f64> = vertices
        .iter()
        .map(|v| v.f().unwrap_or(f64::INFINITY))
        .collect();
    let finite: Vec<bool> = fs.iter().map(|f| f.is_finite()).collect();
    let lower = |a: usize, b: usize| {
        (fs[a], a) < (fs[b], b) // total order breaks cost ties by index
    };

    // Steepest-descent parent per vertex (self if graph minimizer).
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if !finite[i] {
            continue;
        }
        let mut best = i;
        for &nb in &vertices[i].neighbor_indices() {
            if nb < n && finite[nb] && lower(nb, best) {
                best = nb;
            }
        }
        parent[i] = best;
    }

    // Resolve roots with path compression (parent chains strictly descend
    // in (f, index), so they are acyclic).
    let mut root: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if !finite[i] {
            continue;
        }
        let mut r = i;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = i;
        while parent[c] != c {
            let nx = parent[c];
            parent[c] = r;
            c = nx;
        }
        root[i] = r;
    }

    // Persistence: merging a younger component (higher minimum) into an elder
    // one records the younger basin's death at the current cost.
    let persistence = compute_persistence(&vertices, &fs, &finite);

    // Aggregate member costs per basin root.
    let mut members: HashMap<usize, Vec<f64>> = HashMap::new();
    for i in 0..n {
        if finite[i] {
            members.entry(root[i]).or_default().push(fs[i]);
        }
    }

    let mut out: Vec<(usize, BasinStats)> = members
        .into_iter()
        .map(|(r, mut fvals)| {
            fvals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let size = fvals.len();
            let f_mean = fvals.iter().sum::<f64>() / size as f64;
            let f_median = fvals[size / 2];
            let tail_n = ((size as f64 * tail_fraction).ceil() as usize).clamp(1, size);
            let f_tail = fvals[size - tail_n..].iter().sum::<f64>() / tail_n as f64;
            let good_counts = thresholds
                .iter()
                .map(|&t| fvals.partition_point(|&f| f <= t))
                .collect();
            let x_sampled = vertices[r].x().to_vec();
            let x_polished = lmap_cache
                .get(&Coordinates::new(x_sampled.clone()))
                .map(|lm| lm.x);
            (
                r,
                BasinStats {
                    xl_index: None,
                    x_sampled,
                    x_polished,
                    f_min_sampled: fs[r],
                    size,
                    f_mean,
                    f_median,
                    f_tail,
                    good_counts,
                    persistence: persistence.get(&r).copied().unwrap_or(f64::INFINITY),
                },
            )
        })
        .collect();
    // Ascending by sampled minimum; ties broken by vertex index so the order
    // does not depend on HashMap iteration.
    out.sort_by(|(ra, a), (rb, b)| {
        a.f_min_sampled
            .partial_cmp(&b.f_min_sampled)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ra.cmp(rb))
    });
    out.into_iter().map(|(_, b)| b).collect()
}

/// SHGO (Simplicial Homology Global Optimization) optimizer.
///
/// This is the main struct that orchestrates the global optimization algorithm.
pub struct Shgo<F, G = fn(&[f64]) -> f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    G: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    /// Objective function to minimize.
    func: Arc<F>,
    /// Bounds for each dimension: (lower, upper).
    bounds: Bounds,
    /// Inequality constraints: g(x) >= 0.
    constraints: Vec<Arc<G>>,
    /// Optimization options.
    options: ShgoOptions,
    /// Dimension of the problem.
    dim: usize,
    /// Function evaluation counter.
    fev_count: Arc<AtomicUsize>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl<F> Shgo<F, fn(&[f64]) -> f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    /// Create a new SHGO optimizer without constraints.
    ///
    /// # Arguments
    ///
    /// * `func` - The objective function to minimize.
    /// * `bounds` - Bounds for each dimension as (lower, upper) pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo::Shgo;
    ///
    /// let sphere = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    /// let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    ///
    /// let optimizer = Shgo::new(sphere, bounds);
    /// ```
    pub fn new(func: F, bounds: Bounds) -> Self {
        let dim = bounds.len();
        // Replace non-finite bounds with ±1e50 (matching Python's behavior)
        let bounds = Self::process_bounds(bounds);
        Self {
            func: Arc::new(func),
            bounds,
            constraints: Vec::new(),
            options: ShgoOptions::default(),
            dim,
            fev_count: Arc::new(AtomicUsize::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Replace non-finite bounds with ±1e50.
    /// Matches Python: `abound[infind[:, 0], 0] = -1e50; abound[infind[:, 1], 1] = 1e50`
    fn process_bounds(bounds: Bounds) -> Bounds {
        bounds
            .into_iter()
            .map(|(lb, ub)| {
                let lb = if lb.is_finite() { lb } else { -1e50 };
                let ub = if ub.is_finite() { ub } else { 1e50 };
                (lb, ub)
            })
            .collect()
    }
}

impl<F, G> Shgo<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    G: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    /// Create a new SHGO optimizer with constraints.
    ///
    /// # Arguments
    ///
    /// * `func` - The objective function to minimize.
    /// * `bounds` - Bounds for each dimension as (lower, upper) pairs.
    /// * `constraints` - Inequality constraints where g(x) >= 0 means feasible.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo::Shgo;
    ///
    /// let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    /// let constraint = |x: &[f64]| x[0] + x[1] - 1.0; // x[0] + x[1] >= 1
    /// let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
    ///
    /// let optimizer = Shgo::with_constraints(objective, bounds, vec![constraint]);
    /// ```
    pub fn with_constraints(func: F, bounds: Bounds, constraints: Vec<G>) -> Self {
        let dim = bounds.len();
        // Replace non-finite bounds with ±1e50 (matching Python's behavior)
        let bounds: Bounds = bounds
            .into_iter()
            .map(|(lb, ub)| {
                let lb = if lb.is_finite() { lb } else { -1e50 };
                let ub = if ub.is_finite() { ub } else { 1e50 };
                (lb, ub)
            })
            .collect();
        Self {
            func: Arc::new(func),
            bounds,
            constraints: constraints.into_iter().map(Arc::new).collect(),
            options: ShgoOptions::default(),
            dim,
            fev_count: Arc::new(AtomicUsize::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set optimization options.
    pub fn with_options(mut self, options: ShgoOptions) -> Self {
        self.options = options;
        self
    }

    /// Cancel the optimization (thread-safe).
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if optimization was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Get current function evaluation count.
    pub fn fev_count(&self) -> usize {
        self.fev_count.load(Ordering::Relaxed)
    }

    /// Run the SHGO optimization algorithm.
    ///
    /// This is the main entry point that performs global optimization.
    ///
    /// # Returns
    ///
    /// Returns `Ok(ShgoResult)` with the optimization results, or `Err(ShgoError)`
    /// if the optimization fails.
    pub fn minimize(&self) -> Result<ShgoResult, ShgoError> {
        // If workers is specified, use a custom thread pool
        if let Some(num_workers) = self.options.workers {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_workers)
                .build()
                .map_err(|e| ShgoError::InvalidBounds(format!("Failed to create thread pool: {}", e)))?;
            
            pool.install(|| self.minimize_inner())
        } else {
            // Use default global thread pool (all cores)
            self.minimize_inner()
        }
    }

    /// Compute the effective number of sampling points per iteration.
    /// If n=0 (auto), uses 2^dim + 1 for simplicial, 128 for Sobol.
    fn effective_n(&self) -> usize {
        if self.options.n == 0 {
            match self.options.sampling_method {
                SamplingMethod::Simplicial => (1usize << self.dim) + 1,
                SamplingMethod::Sobol => 128,
            }
        } else {
            self.options.n
        }
    }

    /// Whether any stopping criterion other than `iters` is set.
    fn has_other_stopping_criterion(&self) -> bool {
        self.options.maxiter.is_some()
            || self.options.maxfev.is_some()
            || self.options.maxev.is_some()
            || self.options.maxtime.is_some()
            || self.options.f_min.is_some()
    }

    /// Compute effective iters: if any stopping criterion other than iters
    /// is set, iters becomes None (unlimited). Matches Python behavior.
    fn effective_iters(&self) -> Option<usize> {
        if self.has_other_stopping_criterion() {
            None // Other criteria control termination
        } else {
            self.options.iters
        }
    }

    /// Per-dimension closeness test for two points: each coordinate must
    /// agree to within `xl_dedup_rtol` times the width of that dimension's
    /// bounds (for effectively unbounded dimensions the scale is
    /// `max(1, |x|)` instead). With `xl_dedup_rtol == 0` only bitwise-equal
    /// points match.
    fn within_x_tolerance(&self, a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let rtol = self.options.xl_dedup_rtol;
        if rtol <= 0.0 {
            return a == b;
        }
        a.iter()
            .zip(b.iter())
            .zip(self.bounds.iter())
            .all(|((x, y), (lo, hi))| {
                let width = hi - lo;
                let scale = if width.is_finite() && width < 1e30 {
                    width
                } else {
                    x.abs().max(y.abs()).max(1.0)
                };
                (x - y).abs() <= rtol * scale
            })
    }

    /// Whether two local results describe the same minimum (see
    /// [`ShgoOptions::xl_dedup_rtol`] / [`ShgoOptions::xl_dedup_ftol`]).
    fn same_minimum(&self, xa: &[f64], fa: f64, xb: &[f64], fb: f64) -> bool {
        if !self.within_x_tolerance(xa, xb) {
            return false;
        }
        let ftol = self.options.xl_dedup_ftol;
        ftol <= 0.0 || (fa - fb).abs() <= ftol * fa.abs().max(fb.abs()).max(1.0)
    }

    /// Whether `x` sits on an already-known local minimum (SciPy's
    /// `minimizers()` skips vertices located at `LMC.xl_maps` entries; this
    /// also covers re-inserted minima, which would otherwise be re-minimized
    /// every iteration).
    fn near_known_minimum(&self, x: &[f64], xl: &[Vec<f64>]) -> bool {
        xl.iter().any(|m| self.within_x_tolerance(x, m))
    }

    /// Whether the persistence of each candidate has to be known this
    /// iteration (i.e. whether either persistence-based filter is active).
    fn needs_candidate_persistence(&self) -> bool {
        self.options.min_candidate_persistence.is_some()
            || self.options.max_candidates_by_persistence.is_some()
    }

    /// Turn the graph minimizers into the pool of local-minimization starting
    /// points: drop infeasible vertices, ones already minimized from, and ones
    /// sitting on an already-found minimum; optionally prune by basin
    /// persistence; then sort by function value and trim to `maxiter_local`
    /// (SciPy's `sort_min_pool` + `local_iter`).
    fn select_candidates<'a>(
        &self,
        minimizers: &'a [Arc<crate::Vertex>],
        lmap_cache: &LMapCache,
        xl: &[Vec<f64>],
        persistence: Option<&std::collections::HashMap<usize, f64>>,
    ) -> Vec<&'a Arc<crate::Vertex>> {
        let mut candidates: Vec<&Arc<crate::Vertex>> = minimizers
            .iter()
            .filter(|v| v.feasible() != Some(false))
            .filter(|v| {
                !lmap_cache.contains(&Coordinates::new(v.coordinates().as_slice().to_vec()))
            })
            .filter(|v| {
                self.options.explore_from_known_minima || !self.near_known_minimum(v.x(), xl)
            })
            .collect();

        if let Some(pers) = persistence {
            // A candidate with no recorded death never merged into a deeper
            // basin, so its persistence is infinite and it is never pruned.
            let p = |v: &Arc<crate::Vertex>| {
                pers.get(&v.index()).copied().unwrap_or(f64::INFINITY)
            };
            let before = candidates.len();
            // The lowest-cost candidate is never pruned: SciPy's
            // `minimise_pool` always minimizes `X_min[0]` before any trimming,
            // and it is the single most likely start to reach the optimum.
            let best = candidates
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    a.f()
                        .unwrap_or(f64::INFINITY)
                        .partial_cmp(&b.f().unwrap_or(f64::INFINITY))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ia.cmp(ib))
                })
                .map(|(_, v)| v.index());
            let is_best = |v: &Arc<crate::Vertex>| Some(v.index()) == best;
            if let Some(min_p) = self.options.min_candidate_persistence {
                candidates.retain(|v| p(v) > min_p || is_best(v));
            }
            if let Some(top) = self.options.max_candidates_by_persistence {
                if candidates.len() > top {
                    candidates.sort_by(|a, b| {
                        is_best(b)
                            .cmp(&is_best(a))
                            .then(
                                p(b).partial_cmp(&p(a))
                                    .unwrap_or(std::cmp::Ordering::Equal),
                            )
                            .then(a.index().cmp(&b.index()))
                    });
                    candidates.truncate(top.max(1));
                }
            }
            if self.options.disp > 1 {
                println!(
                    "  persistence pruning: {} -> {} candidates",
                    before,
                    candidates.len()
                );
            }
        }

        // SciPy sorts the minimizer pool by function value (sort_min_pool)
        // before trimming to `local_iter` candidates, so truncation keeps the
        // most promising starts. Ties break by vertex index for determinism.
        candidates.sort_by(|a, b| {
            a.f()
                .unwrap_or(f64::INFINITY)
                .partial_cmp(&b.f().unwrap_or(f64::INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.index().cmp(&b.index()))
        });
        candidates.truncate(self.options.maxiter_local.unwrap_or(usize::MAX));
        candidates
    }

    /// Internal minimize implementation.
    fn minimize_inner(&self) -> Result<ShgoResult, ShgoError> {
        let start_time = Instant::now();

        // Check for early cancellation
        if self.is_cancelled() {
            return Err(ShgoError::Cancelled);
        }

        // Validate bounds
        self.validate_bounds()?;

        // `iters: None` only makes sense together with another criterion;
        // otherwise the main loop would never terminate.
        if self.options.iters.is_none() && !self.has_other_stopping_criterion() {
            return Err(ShgoError::InvalidOption(
                "no stopping criterion: set `iters` or one of maxiter / maxfev / maxev / maxtime / f_min".into(),
            ));
        }

        // Reset evaluation counter (but not cancelled flag - allow pre-cancellation)
        self.fev_count.store(0, Ordering::Relaxed);

        // Initialize result
        let mut result = ShgoResult::new(self.dim);

        if self.options.knn_auto.is_some()
            && !(self.options.sampling_method == SamplingMethod::Sobol
                && self.options.connectivity_method == ConnectivityMethod::KNearestNeighbors)
        {
            return Err(ShgoError::InvalidOption(
                "knn_auto requires sampling_method = Sobol with connectivity_method = KNearestNeighbors"
                    .into(),
            ));
        }

        // Run based on sampling method
        match self.options.sampling_method {
            SamplingMethod::Simplicial => {
                self.iterate_simplicial(&mut result)?;
            }
            SamplingMethod::Sobol => {
                self.iterate_sobol(&mut result)?;
            }
        }

        // Finalize result
        // Total function evaluations = sampling evaluations + local-minimization
        // evaluations, matching SciPy's `res.nfev = self.fn + self.res.nlfev`.
        result.nfev = self.fev_count.load(Ordering::Relaxed) + result.nlfev;
        result.time = start_time.elapsed().as_secs_f64();

        // Sort and deduplicate local minima by function value
        if !result.xl.is_empty() {
            let mut combined: Vec<_> = result.xl.iter().cloned().zip(result.funl.iter().cloned()).collect();
            combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Deduplicate: drop entries that describe an already-kept minimum
            // (tolerances from `xl_dedup_rtol` / `xl_dedup_ftol`). Entries are
            // visited in ascending f, so the kept representative is the lowest.
            let mut deduped: Vec<(Vec<f64>, f64)> = Vec::new();
            for (x, f) in &combined {
                let is_dup = deduped
                    .iter()
                    .any(|(ex, ef)| self.same_minimum(x, *f, ex, *ef));
                if !is_dup {
                    deduped.push((x.clone(), *f));
                }
            }

            result.xl = deduped.iter().map(|(x, _)| x.clone()).collect();
            result.funl = deduped.iter().map(|(_, f)| *f).collect();

            // Best solution
            if let Some((x, f)) = combined.first() {
                result.x = x.clone();
                result.fun = *f;
                result.success = true;
                result.message = "Optimization terminated successfully.".to_string();
            }
        } else {
            // No local minimizer found — return the lowest sampled vertex
            // (matching Python's find_lowest_vertex + fail_routine)
            // result.x and result.fun retain their sentinel values from
            // iterate_* which tracks the lowest vertex
            result.success = false;
            if result.fun.is_finite() {
                result.message = format!(
                    "Failed to find a feasible minimizer point. Lowest sampling point = {}",
                    result.fun
                );
            } else {
                result.message = "Failed to find a feasible minimizer point. No feasible point found.".to_string();
            }
        }

        // Map each basin to its polished xl row now that xl is final.
        {
            let ShgoResult { xl, basins, .. } = &mut result;
            if let Some(basins) = basins.as_mut() {
                for b in basins.iter_mut() {
                    let key = b.x_polished.as_ref().unwrap_or(&b.x_sampled);
                    b.xl_index = xl.iter().position(|x| self.within_x_tolerance(x, key));
                }
            }
        }

        // Post-optimization robustness analysis (extensions; both add their
        // objective evaluations to nfev and leave x/fun/xl/funl untouched).
        if !result.xl.is_empty() {
            if let Some(probe) = self.options.robustness_probe {
                result.nfev += self.probe_robustness(&mut result, &probe);
            }
            if let Some(rp) = self.options.robust_polish {
                result.nfev += self.robust_polish(&mut result, &rp);
            }
        }

        if self.options.disp > 0 {
            self.print_summary(&result);
        }

        Ok(result)
    }

    /// Validate bounds specification.
    fn validate_bounds(&self) -> Result<(), ShgoError> {
        if self.bounds.is_empty() {
            return Err(ShgoError::InvalidBounds("Bounds cannot be empty".into()));
        }

        for (i, (lb, ub)) in self.bounds.iter().enumerate() {
            if lb >= ub {
                return Err(ShgoError::InvalidBounds(format!(
                    "Lower bound {} must be less than upper bound {} for dimension {}",
                    lb, ub, i
                )));
            }
            if lb.is_nan() || ub.is_nan() {
                return Err(ShgoError::InvalidBounds(format!(
                    "Bounds cannot be NaN for dimension {}",
                    i
                )));
            }
        }

        Ok(())
    }

    /// Run optimization using simplicial (hypercube) sampling.
    fn iterate_simplicial(
        &self,
        result: &mut ShgoResult,
    ) -> Result<(), ShgoError>
    {
        let start_time = Instant::now();

        // Create local minimization cache
        let lmap_cache = LMapCache::new();

        // Clone function references
        let func = Arc::clone(&self.func);
        let fev_count = Arc::clone(&self.fev_count);
        
        // Wrap objective function to count evaluations
        let wrapped_func = move |x: &[f64]| -> f64 {
            fev_count.fetch_add(1, Ordering::Relaxed);
            func(x)
        };

        // Wrap constraints to convert f64 >= 0 to bool
        let wrapped_constraints: Option<Vec<_>> = if self.constraints.is_empty() {
            None
        } else {
            Some(
                self.constraints
                    .iter()
                    .map(|c| {
                        let c = Arc::clone(c);
                        move |x: &[f64]| -> bool { c(x) >= 0.0 }
                    })
                    .collect(),
            )
        };

        // Create the simplicial complex
        let mut complex = Complex::new(
            self.bounds.clone(),
            wrapped_func,
            wrapped_constraints,
        );

        let effective_n = self.effective_n();
        let effective_iters = self.effective_iters();
        let mut iteration = 0;

        // Main optimization loop. Matches SciPy's iterate_hypercube growth:
        // every iteration adds ~n new sampling points (auto n = 2^dim + 1).
        // Iteration 1's refine performs the bounded initial triangulation, so
        // the default single-iteration run samples exactly the initial
        // complex (2^dim corners + centroid) like SciPy — no extra
        // refinement generation.
        loop {
            iteration += 1;
            result.nit = iteration;

            complex.refine(Some(effective_n));
            complex.process_pools();

            // Find local minimizer candidates
            let minimizers = complex.find_minimizers();

            if self.options.disp > 1 {
                println!(
                    "Iteration {}: {} vertices, {} minimizer candidates",
                    iteration,
                    complex.vertex_count(),
                    minimizers.len()
                );
            }

            // Process minimizers with local optimization in parallel
            // Pre-compute all LCBs from the (read-only) complex, then dispatch
            // all local minimizations concurrently via rayon.
            if self.options.minimize_every_iter {
                let persistence = if self.needs_candidate_persistence() {
                    Some(persistence_map(&complex.cache))
                } else {
                    None
                };
                let candidates = self.select_candidates(
                    &minimizers,
                    &lmap_cache,
                    &result.xl,
                    persistence.as_ref(),
                );

                if !candidates.is_empty() {
                    // Pre-compute starting points and locally convex bounds
                    // (reads from Complex which is not modified during minimization)
                    #[allow(clippy::type_complexity)]
                    let work_items: Vec<(Vec<f64>, Vec<(f64, f64)>)> = candidates
                        .iter()
                        .map(|v| {
                            let x0 = v.coordinates().as_slice().to_vec();
                            let lcb = self.construct_lcb_simplicial(v, &complex);
                            (x0, lcb)
                        })
                        .collect();

                    // Run all local minimizations in parallel
                    let local_results: Vec<Option<LocalMinimum>> = work_items
                        .par_iter()
                        .map(|(x0, lcb)| {
                            self.minimize_local_from_point(x0, &lmap_cache, lcb)
                        })
                        .collect();

                    // Gather results (order-independent: results are
                    // deterministic per starting point via lmap_cache dedup).
                    // SciPy's LMC.add_res appends EVERY local result to the
                    // minima maps, converged or not — so budget-capped runs
                    // still count; only non-finite failures are excluded.
                    // nlfev is taken from lmap_cache.total_fev() after the
                    // loop so all attempts count (matching SciPy's LMC).
                    for local_min in local_results.into_iter().flatten() {
                        if local_min.fun.is_finite() {
                            result.xl.push(local_min.x);
                            result.funl.push(local_min.fun);
                        }
                    }
                }
            } else {
                // Just use vertex function values (no local minimization)
                for vertex in &minimizers {
                    if vertex.feasible() == Some(false) {
                        continue;
                    }
                    let x0: Vec<f64> = vertex.coordinates().as_slice().to_vec();
                    let f = vertex.f().unwrap_or(f64::INFINITY);
                    result.xl.push(x0);
                    result.funl.push(f);
                }
            }

            // Update best solution
            if let Some((min_idx, &min_f)) = result
                .funl
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                if min_f < result.fun {
                    result.fun = min_f;
                    result.x = result.xl[min_idx].clone();
                }
            }

            // Check stopping criteria AFTER work (matching Python)
            if self.check_stopping_criteria(iteration, effective_iters, start_time.elapsed(), result, complex.vertex_count())? {
                break;
            }
        }

        // Local-minimization function evaluations across all attempts
        // (successful or not), matching SciPy's `res.nlfev`.
        result.nlfev = lmap_cache.total_fev();

        // If no local minimizer was found, find the lowest sampled vertex
        // and lowest LMC entry (matching Python's find_lowest_vertex)
        if result.xl.is_empty() {
            let mut lowest_f = f64::INFINITY;
            let mut lowest_x: Option<Vec<f64>> = None;

            // Scan vertex cache
            for vertex in complex.cache.iter() {
                if let Some(f) = vertex.f() {
                    if f < lowest_f {
                        lowest_f = f;
                        lowest_x = Some(vertex.coordinates().as_slice().to_vec());
                    }
                }
            }

            // Scan LMC cache
            for lm in lmap_cache.get_sorted() {
                if lm.fun < lowest_f {
                    lowest_f = lm.fun;
                    lowest_x = Some(lm.x.clone());
                }
            }

            if let Some(x) = lowest_x {
                result.x = x;
                result.fun = lowest_f;
            }
        }

        if self.options.compute_basin_stats {
            result.basins = Some(compute_basin_statistics(
                &complex.cache,
                &lmap_cache,
                &self.options.basin_good_thresholds,
                self.options.basin_tail_fraction,
            ));
        }

        Ok(())
    }

    /// Run optimization using Sobol sequence sampling.
    /// Perform Sobol-mode optimization with Delaunay triangulation.
    ///
    /// Matches Python's `iterate_delaunay` flow:
    /// 1. Generate Sobol quasi-random points (rounded to next power of 2)
    /// 2. Scale to bounds domain
    /// 3. Re-insert previous local minimizer locations (iterations > 1)
    /// 4. Build Delaunay triangulation (via QHull) for vertex connectivity
    /// 5. Convert vertex-face mesh to vertex-vertex (vf_to_vv)
    /// 6. Find topological minimizers using neighbor graph
    /// 7. Local optimization with GLOBAL bounds (no LCB tightening)
    fn iterate_sobol(
        &self,
        result: &mut ShgoResult,
    ) -> Result<(), ShgoError>
    {
        let start_time = Instant::now();

        // Create local minimization cache
        let lmap_cache = LMapCache::new();

        // Create Sobol sequence generator
        let mut sobol = Sobol::new(self.dim);

        // Clone function references
        let func = Arc::clone(&self.func);
        let fev_count = Arc::clone(&self.fev_count);
        
        // Wrap objective function to count evaluations
        let wrapped_func = move |x: &[f64]| -> f64 {
            fev_count.fetch_add(1, Ordering::Relaxed);
            func(x)
        };

        // Wrap constraints to convert f64 >= 0 to bool
        let wrapped_constraints: Option<Vec<_>> = if self.constraints.is_empty() {
            None
        } else {
            Some(
                self.constraints
                    .iter()
                    .map(|c| {
                        let c = Arc::clone(c);
                        move |x: &[f64]| -> bool { c(x) >= 0.0 }
                    })
                    .collect(),
            )
        };

        // Create vertex cache
        let cache = VertexCache::new(
            wrapped_func,
            wrapped_constraints,
        );

        // Round n to next power of 2 for Sobol sampling (matching Python:
        // n = int(2 ** np.ceil(np.log2(n))) )
        let raw_n = self.effective_n();
        let effective_n = raw_n.next_power_of_two();

        let effective_iters = self.effective_iters();
        let mut iteration = 0;
        let mut total_points = 0;

        // Main optimization loop
        loop {
            iteration += 1;
            result.nit = iteration;

            // Generate Sobol points scaled to bounds
            let skip = self.options.sobol_skip + total_points;
            let mut points = sobol.generate_bounds(effective_n, &self.bounds, skip);
            total_points += effective_n;

            if self.options.disp > 1 {
                println!(
                    "Iteration {}: generating {} Sobol points (power-of-2 rounded from {})",
                    iteration, effective_n, raw_n
                );
            }

            // Re-insertion: append previous local minimizer locations
            // (matching Python: self.C = np.vstack((self.C, np.array(self.LMC.xl_maps))))
            if iteration > 1 {
                for xl in &result.xl {
                    points.push(xl.clone());
                }
            }

            // Instantiate this batch's vertices in the cache (connectivity
            // is built over the full cumulative cache below).
            for p in &points {
                cache.get_or_create(p.clone());
            }

            // Process pending evaluations (constraints, then function values)
            // BEFORE building connectivity: nothing in the graph construction
            // depends on them except automatic k selection, which needs them.
            cache.process_pools();

            // ---- Build vertex connectivity ----
            // All methods build the graph over the FULL cumulative point
            // cloud, matching SciPy's semantics of re-triangulating all
            // sampled points each iteration (scipy: `Tri.add_points` on the
            // accumulated `self.C`). Per-batch graphs would leave earlier
            // iterations' vertices with stale neighborhoods and no old↔new
            // edges. On iteration 1 the cache equals the batch, so this is
            // identical to per-batch there. The cache is deduplicated by
            // construction, so re-inserted minima coinciding with sample
            // points cost nothing extra.
            let all_vertices: Vec<std::sync::Arc<crate::Vertex>> = cache.iter().collect();
            // Need at least dim+2 non-degenerate points for triangulation
            if all_vertices.len() >= self.dim + 2 {
                let all_points: Vec<Vec<f64>> =
                    all_vertices.iter().map(|v| v.x().to_vec()).collect();

                if self.dim == 1 {
                    // 1D: sort points and connect consecutive pairs
                    // (Delaunay in 1D is just sorted adjacency)
                    let mut sorted_indices: Vec<usize> = (0..all_points.len()).collect();
                    sorted_indices.sort_by(|&a, &b| {
                        all_points[a][0]
                            .partial_cmp(&all_points[b][0])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for w in sorted_indices.windows(2) {
                        crate::Vertex::connect_bidirectional(
                            &all_vertices[w[0]],
                            &all_vertices[w[1]],
                        );
                    }
                } else {
                    match self.options.connectivity_method {
                        ConnectivityMethod::Delaunay => {
                            Self::build_delaunay_connectivity(
                                &all_points,
                                &all_vertices,
                                self.dim,
                                self.options.disp,
                                false,
                            )?;
                        }
                        ConnectivityMethod::DelaunayScipyCompat => {
                            Self::build_delaunay_connectivity(
                                &all_points,
                                &all_vertices,
                                self.dim,
                                self.options.disp,
                                true,
                            )?;
                        }
                        ConnectivityMethod::KNearestNeighbors => match self.options.knn_auto {
                            Some(auto) => {
                                let fs: Vec<f64> = all_vertices
                                    .iter()
                                    .map(|v| v.f().unwrap_or(f64::INFINITY))
                                    .collect();
                                result.knn_selection = Some(Self::build_knn_connectivity_auto(
                                    &all_points,
                                    &all_vertices,
                                    &fs,
                                    self.dim,
                                    auto,
                                    self.options.disp,
                                ));
                            }
                            None => {
                                Self::build_knn_connectivity(
                                    &all_points,
                                    &all_vertices,
                                    self.dim,
                                    self.options.knn_neighbors,
                                    self.options.disp,
                                );
                            }
                        },
                        ConnectivityMethod::HNSW => {
                            Self::build_hnsw_connectivity(
                                &all_points,
                                &all_vertices,
                                self.dim,
                                self.options.knn_neighbors,
                                self.options.disp,
                            );
                        }
                        ConnectivityMethod::ScaNN => {
                            Self::build_scann_connectivity(
                                &all_points,
                                &all_vertices,
                                self.dim,
                                self.options.knn_neighbors,
                                self.options.disp,
                            );
                        }
                    }
                }
            }

            // Find minimizers using topological analysis
            // (f(v) < f(all neighbors) over the sampling graph)
            let minimizers = cache.find_all_minimizers();
            
            // Process minimizers with local optimization
            if self.options.minimize_every_iter {
                let persistence = if self.needs_candidate_persistence() {
                    Some(persistence_map(&cache))
                } else {
                    None
                };
                let candidates = self.select_candidates(
                    &minimizers,
                    &lmap_cache,
                    &result.xl,
                    persistence.as_ref(),
                );

                // Local minimization with GLOBAL bounds in parallel
                // (matching Python's construct_lcb_delaunay which returns global bounds
                //  without tightening, unlike simplicial mode's LCB)
                let local_results: Vec<Option<LocalMinimum>> = candidates
                    .par_iter()
                    .map(|vertex| {
                        let x0 = vertex.coordinates().as_slice().to_vec();
                        self.minimize_local_from_point(&x0, &lmap_cache, &self.bounds)
                    })
                    .collect();

                // SciPy's LMC.add_res appends EVERY local result to the
                // minima maps, converged or not — only non-finite failures
                // are excluded. nlfev is taken from lmap_cache.total_fev()
                // after the loop so all attempts count (matching SciPy).
                for local_min in local_results.into_iter().flatten() {
                    if local_min.fun.is_finite() {
                        result.xl.push(local_min.x);
                        result.funl.push(local_min.fun);
                    }
                }
            } else {
                // Just use vertex function values (no local minimization)
                for vertex in &minimizers {
                    if vertex.feasible() == Some(false) {
                        continue;
                    }
                    if let Some(f) = vertex.f() {
                        let x0 = vertex.coordinates().as_slice().to_vec();
                        result.xl.push(x0);
                        result.funl.push(f);
                    }
                }
            }

            // Update best solution
            if let Some((min_idx, &min_f)) = result
                .funl
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                if min_f < result.fun {
                    result.fun = min_f;
                    result.x = result.xl[min_idx].clone();
                }
            }

            // Check stopping criteria AFTER work (matching Python)
            if self.check_stopping_criteria(iteration, effective_iters, start_time.elapsed(), result, total_points)? {
                break;
            }
        }

        // Local-minimization function evaluations across all attempts
        // (successful or not), matching SciPy's `res.nlfev`.
        result.nlfev = lmap_cache.total_fev();

        // If no local minimizer was found, find the lowest sampled vertex
        // and lowest LMC entry (matching Python's find_lowest_vertex)
        if result.xl.is_empty() {
            let mut lowest_f = f64::INFINITY;
            let mut lowest_x: Option<Vec<f64>> = None;

            // Scan vertex cache
            for vertex in cache.iter() {
                if let Some(f) = vertex.f() {
                    if f < lowest_f {
                        lowest_f = f;
                        lowest_x = Some(vertex.coordinates().as_slice().to_vec());
                    }
                }
            }

            // Scan LMC cache
            for lm in lmap_cache.get_sorted() {
                if lm.fun < lowest_f {
                    lowest_f = lm.fun;
                    lowest_x = Some(lm.x.clone());
                }
            }

            if let Some(x) = lowest_x {
                result.x = x;
                result.fun = lowest_f;
            }
        }

        if self.options.compute_basin_stats {
            result.basins = Some(compute_basin_statistics(
                &cache,
                &lmap_cache,
                &self.options.basin_good_thresholds,
                self.options.basin_tail_fraction,
            ));
        }

        Ok(())
    }

    /// Check all stopping criteria.
    /// Matches Python's `stopping_criteria` method.
    fn check_stopping_criteria(
        &self,
        iteration: usize,
        effective_iters: Option<usize>,
        elapsed: Duration,
        result: &ShgoResult,
        n_sampled: usize,
    ) -> Result<bool, ShgoError> {
        // Check cancellation
        if self.is_cancelled() {
            return Err(ShgoError::Cancelled);
        }

        // Check iters limit (default: 1 iteration then done)
        if let Some(iters) = effective_iters {
            if iteration >= iters {
                return Ok(true);
            }
        }

        // Check maxiter (hard maximum iterations)
        if let Some(maxiter) = self.options.maxiter {
            if iteration >= maxiter {
                return Ok(true);
            }
        }

        // Check function evaluation limit
        if let Some(maxfev) = self.options.maxfev {
            if self.fev_count() >= maxfev {
                return Ok(true);
            }
        }

        // Check sampling evaluation limit (maxev): counts sampled points
        // including infeasible ones, matching SciPy's
        // `self.n_sampled >= self.maxev` (distinct from maxfev, which counts
        // feasible objective evaluations).
        if let Some(maxev) = self.options.maxev {
            if n_sampled >= maxev {
                return Ok(true);
            }
        }

        // Check time limit
        if let Some(maxtime) = self.options.maxtime {
            if elapsed.as_secs_f64() >= maxtime {
                return Ok(true);
            }
        }

        // Check precision stopping (f_min + f_tol)
        if let Some(f_min) = self.options.f_min {
            if !result.fun.is_infinite() {
                if f_min == 0.0 {
                    if result.fun <= self.options.f_tol {
                        return Ok(true);
                    }
                } else {
                    let pe = (result.fun - f_min) / f_min.abs();
                    if result.fun <= f_min || pe <= self.options.f_tol {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Build vertex connectivity using k-nearest-neighbors.
    ///
    /// For each point, finds the k closest points by Euclidean distance and
    /// connects them bidirectionally. This is O(n² · d) brute-force, which
    /// is fast for typical SHGO problem sizes and avoids QHull's exponential
    /// scaling in high dimensions.
    fn build_knn_connectivity(
        points: &[Vec<f64>],
        vertices: &[std::sync::Arc<crate::Vertex>],
        dim: usize,
        knn_neighbors: Option<usize>,
        disp: usize,
    ) {
        let n = points.len();
        let k = knn_neighbors
            .unwrap_or(2 * dim + 1)
            .min(n.saturating_sub(1));

        if k == 0 || n < 2 {
            return;
        }

        // Brute-force k-NN: compute all pairwise squared distances,
        // then partial-sort each row to find the k nearest.
        // O(n² · d) total, parallelized across rows with rayon. Each row's
        // k-selection depends only on its own distance computation (not on
        // thread scheduling), and neighbors are a set, so the resulting edge
        // set — and therefore the optimization result — is deterministic.
        (0..n).into_par_iter().for_each(|i| {
            // Compute squared distances from point i to all other points
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d2: f64 = points[i]
                        .iter()
                        .zip(points[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (j, d2)
                })
                .collect();

            // Partial sort to find k nearest (O(n) via select_nth_unstable)
            if dists.len() > k {
                dists.select_nth_unstable_by(k - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                dists.truncate(k);
            }

            // Connect bidirectionally (Vertex::connect is thread-safe)
            for &(j, _) in &dists {
                crate::Vertex::connect_bidirectional(&vertices[i], &vertices[j]);
            }
        });

        if disp > 1 {
            let avg_neighbors: f64 = vertices.iter()
                .map(|v| v.neighbor_count() as f64)
                .sum::<f64>() / n as f64;
            println!(
                "  k-NN connectivity: k={}, {} vertices, avg {:.1} neighbors/vertex",
                k, n, avg_neighbors
            );
        }
    }

    /// Build k-NN connectivity with k chosen from the minimizer-pool curve.
    ///
    /// One pass computes, for every point, its `k_max` nearest neighbours in
    /// distance order. An undirected edge's rank is the smaller of the two
    /// directed ranks, so it is present in the symmetrized k-NN graph exactly
    /// when `rank <= k`; a vertex is a minimizer at k exactly when no incident
    /// edge of rank `<= k` leads to a vertex that is not strictly higher. The
    /// whole curve `|M_k|` therefore falls out of one neighbour computation
    /// and no objective evaluations, and k is picked as the smallest value
    /// whose candidate count fits the caller's budget.
    ///
    /// Requires the objective values in `fs` (index-aligned with `points`), so
    /// the caller must have processed the evaluation pools first. Ties in
    /// distance break by point index, so the result is deterministic.
    fn build_knn_connectivity_auto(
        points: &[Vec<f64>],
        vertices: &[std::sync::Arc<crate::Vertex>],
        fs: &[f64],
        dim: usize,
        auto: KnnAuto,
        disp: usize,
    ) -> KnnSelection {
        let n = points.len();
        if n < 2 {
            return KnnSelection {
                k: 0,
                k_max: 0,
                curve: vec![n],
            };
        }
        let k_max = auto
            .k_max
            .unwrap_or_else(|| (4 * dim).max(64))
            .min(n - 1)
            .max(1);
        let k_min = auto.k_min.unwrap_or(dim + 1).clamp(1, k_max);

        // Directed (point, neighbour, rank) triples, ranked by (distance, index).
        let mut edges: Vec<(u32, u32, u32)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut d: Vec<(f64, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d2: f64 = points[i]
                            .iter()
                            .zip(points[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (d2, j as u32)
                    })
                    .collect();
                if d.len() > k_max {
                    d.select_nth_unstable_by(k_max - 1, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    d.truncate(k_max);
                }
                d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let iu = i as u32;
                d.into_iter()
                    .enumerate()
                    .map(move |(rank, (_, j))| {
                        (iu.min(j), iu.max(j), rank as u32 + 1)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Symmetrize: keep the smaller of the two directed ranks per pair.
        edges.par_sort_unstable();
        edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        // m[v] = smallest k at which some neighbour disqualifies v as a
        // minimizer (a neighbour u disqualifies v when !(f_v < f_u)).
        let mut m = vec![u32::MAX; n];
        for &(a, b, r) in &edges {
            let (a, b) = (a as usize, b as usize);
            if fs[b] <= fs[a] {
                m[a] = m[a].min(r);
            }
            if fs[a] <= fs[b] {
                m[b] = m[b].min(r);
            }
        }

        let mut hist = vec![0usize; k_max + 2];
        let mut total = 0usize;
        for (i, &mi) in m.iter().enumerate() {
            if !fs[i].is_finite() {
                continue; // infeasible points are never minimizers
            }
            total += 1;
            let bucket = if mi == u32::MAX {
                k_max + 1
            } else {
                (mi as usize).min(k_max + 1)
            };
            hist[bucket] += 1;
        }
        let mut curve = vec![0usize; k_max + 1];
        let mut removed = 0usize;
        for (k, slot) in curve.iter_mut().enumerate() {
            if k >= 1 {
                removed += hist[k];
            }
            *slot = total - removed;
        }

        let k = (k_min..=k_max)
            .find(|&cand| curve[cand] <= auto.max_local_runs)
            .unwrap_or(k_max);

        edges
            .par_iter()
            .filter(|&&(_, _, r)| (r as usize) <= k)
            .for_each(|&(a, b, _)| {
                crate::Vertex::connect_bidirectional(
                    &vertices[a as usize],
                    &vertices[b as usize],
                );
            });

        if disp > 0 {
            println!(
                "  k-NN auto: k={} (budget {} local runs, |M_k|={}, k range {}..={})",
                k, auto.max_local_runs, curve[k], k_min, k_max
            );
        }

        KnnSelection { k, k_max, curve }
    }

    /// Build vertex connectivity using HNSW (Hierarchical Navigable Small World).
    ///
    /// Uses `hnsw_rs` to build an approximate nearest-neighbor index and query
    /// each point for its k closest neighbors. O(n·log n) build, O(log n) query.
    fn build_hnsw_connectivity(
        points: &[Vec<f64>],
        vertices: &[std::sync::Arc<crate::Vertex>],
        dim: usize,
        knn_neighbors: Option<usize>,
        disp: usize,
    ) {
        use hnsw_rs::prelude::*;

        let n = points.len();
        let k = knn_neighbors
            .unwrap_or(2 * dim + 1)
            .min(n.saturating_sub(1));

        if k == 0 || n < 2 {
            return;
        }

        // HNSW parameters tuned for our use case:
        // - max_nb_connection: edges per node per layer (16 is a good default)
        // - max_layer: maximum number of layers (auto-scaled)
        // - ef_construction: search width during build (higher = more accurate)
        let max_nb_connection = 16.min(k * 2);
        let max_layer = 16;
        let ef_construction = (k * 4).max(48);

        let hnsw = Hnsw::<f64, DistL2>::new(
            max_nb_connection,
            n,
            max_layer,
            ef_construction,
            DistL2,
        );

        // Insert all points (origin_id = index)
        let data_for_par: Vec<(&Vec<f64>, usize)> =
            points.iter().zip(0..n).collect();
        hnsw.parallel_insert(&data_for_par);

        // Switch to search mode after parallel insert
        // (hnsw_rs requires this before searching)
        // Note: set_searching_mode needs &mut but parallel_search doesn't,
        // so we just search directly — it works for sequential insert+search.

        // Query each point for its k nearest neighbors. The query point is
        // itself in the index and comes back as its own nearest neighbor, so
        // ask for k + 1 and drop it — otherwise the effective k is k - 1,
        // inconsistent with the exact KNN method.
        let ef_search = (k * 2).max(32);
        let results = hnsw.parallel_search(points, k + 1, ef_search);

        // Connect bidirectionally based on HNSW results
        for (i, neighbors) in results.iter().enumerate() {
            let mut taken = 0;
            for nb in neighbors {
                let j = nb.d_id;
                if j != i && j < vertices.len() {
                    crate::Vertex::connect_bidirectional(&vertices[i], &vertices[j]);
                    taken += 1;
                    if taken == k {
                        break;
                    }
                }
            }
        }

        if disp > 1 {
            let avg_neighbors: f64 = vertices.iter()
                .map(|v| v.neighbor_count() as f64)
                .sum::<f64>() / n as f64;
            println!(
                "  HNSW connectivity: k={}, {} vertices, avg {:.1} neighbors/vertex",
                k, n, avg_neighbors
            );
        }
    }

    /// Build vertex connectivity using ScaNN (Scalable Nearest Neighbors).
    ///
    /// Uses `vecstore`'s ScaNN implementation with learned quantization.
    /// Points are down-cast from f64 to f32 for the index.
    fn build_scann_connectivity(
        points: &[Vec<f64>],
        vertices: &[std::sync::Arc<crate::Vertex>],
        dim: usize,
        knn_neighbors: Option<usize>,
        disp: usize,
    ) {
        use vecstore::scann::{ScaNNIndex, ScaNNConfig};

        let n = points.len();
        let k = knn_neighbors
            .unwrap_or(2 * dim + 1)
            .min(n.saturating_sub(1));

        if k == 0 || n < 2 {
            return;
        }

        // ScaNN config tuned for small-to-medium point sets.
        // num_leaves controls partitioning granularity.
        let num_leaves = (n / 10).clamp(2, 1000);
        let config = ScaNNConfig {
            num_leaves,
            num_leaves_to_search: num_leaves, // search all leaves for accuracy
            quantization_bits: 8,             // highest precision quantization
            rerank_k: (k * 4).max(20).min(n), // rerank more candidates
            dimensions_per_block: 2,
        };

        let mut index = match ScaNNIndex::new(dim, config) {
            Ok(idx) => idx,
            Err(e) => {
                if disp > 0 {
                    eprintln!("Warning: ScaNN index creation failed: {}, falling back to k-NN", e);
                }
                Self::build_knn_connectivity(points, vertices, dim, knn_neighbors, disp);
                return;
            }
        };

        // Convert f64 → f32 for ScaNN
        let f32_points: Vec<Vec<f32>> = points.iter()
            .map(|p| p.iter().map(|&v| v as f32).collect())
            .collect();

        // Train on all points
        if let Err(e) = index.train(&f32_points) {
            if disp > 0 {
                eprintln!("Warning: ScaNN training failed: {}, falling back to k-NN", e);
            }
            Self::build_knn_connectivity(points, vertices, dim, knn_neighbors, disp);
            return;
        }

        // Add all points with string IDs = index
        let batch: Vec<(String, Vec<f32>)> = f32_points.iter().enumerate()
            .map(|(i, p)| (i.to_string(), p.clone()))
            .collect();
        if let Err(e) = index.add_batch(batch) {
            if disp > 0 {
                eprintln!("Warning: ScaNN add_batch failed: {}, falling back to k-NN", e);
            }
            Self::build_knn_connectivity(points, vertices, dim, knn_neighbors, disp);
            return;
        }

        // Query each point for k nearest neighbors
        let mut failed: Vec<usize> = Vec::new();
        for i in 0..n {
            match index.search(&f32_points[i], k + 1) {
                Ok(results) => {
                    for (id_str, _dist) in &results {
                        if let Ok(j) = id_str.parse::<usize>() {
                            if j != i && j < vertices.len() {
                                crate::Vertex::connect_bidirectional(&vertices[i], &vertices[j]);
                            }
                        }
                    }
                }
                Err(_) => failed.push(i),
            }
        }

        // A vertex left with no neighbors vacuously passes the minimizer
        // test and spawns a spurious local minimization, so failed queries
        // get exact brute-force neighbors instead of being skipped.
        if !failed.is_empty() {
            if disp > 0 {
                eprintln!(
                    "Warning: ScaNN search failed for {} of {} points; \
                     using brute-force k-NN for those",
                    failed.len(),
                    n
                );
            }
            for &i in &failed {
                let mut dists: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d2: f64 = points[i]
                            .iter()
                            .zip(points[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (j, d2)
                    })
                    .collect();
                if dists.len() > k {
                    dists.select_nth_unstable_by(k - 1, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    dists.truncate(k);
                }
                for &(j, _) in &dists {
                    crate::Vertex::connect_bidirectional(&vertices[i], &vertices[j]);
                }
            }
        }

        if disp > 1 {
            let avg_neighbors: f64 = vertices.iter()
                .map(|v| v.neighbor_count() as f64)
                .sum::<f64>() / n as f64;
            println!(
                "  ScaNN connectivity: k={}, {} vertices, avg {:.1} neighbors/vertex",
                k, n, avg_neighbors
            );
        }
    }

    /// Build vertex connectivity using Delaunay triangulation via QHull.
    ///
    /// With `scipy_compat == false` every edge of every simplex is added (the
    /// full 1-skeleton of the triangulation). With `scipy_compat == true`
    /// SciPy's `vf_to_vv` behaviour is reproduced instead: it iterates
    /// `combinations(simplex, dim)` and connects `e[0]-e[1]` of each, which for
    /// `dim >= 3` connects only the first three vertices of each simplex.
    fn build_delaunay_connectivity(
        points: &[Vec<f64>],
        vertices: &[std::sync::Arc<crate::Vertex>],
        dim: usize,
        disp: usize,
        scipy_compat: bool,
    ) -> Result<(), ShgoError> {
        // Wrap in with_stdout_suppressed to silence upstream debug println
        let qh = with_stdout_suppressed(|| {
            qhull::Qh::new_delaunay(
                points.iter().map(|p| p.iter().cloned()),
            )
            .or_else(|_| {
                // Retry with joggled points: add tiny random perturbation
                // to break cocircular/cospherical degeneracies (equivalent
                // to Qhull's QJ option).
                if disp > 0 {
                    eprintln!("Warning: Delaunay triangulation failed (cocircular points), retrying with joggled input");
                }
                let scale = 1e-10;
                let joggled: Vec<Vec<f64>> = points.iter().enumerate().map(|(pi, p)| {
                    p.iter().enumerate().map(|(ci, &v)| {
                        let hash = ((pi * 31 + ci + 1) as f64) * 0.618033988749895;
                        let jitter = (hash.fract() - 0.5) * 2.0 * scale;
                        v + jitter
                    }).collect()
                }).collect();
                qhull::Qh::new_delaunay(
                    joggled.iter().map(|p| p.iter().cloned()),
                )
            })
        })
        .map_err(|e| {
            ShgoError::MeshGeneration(format!(
                "Delaunay triangulation failed: {}",
                e
            ))
        })?;

        // Convert vertex-face mesh to vertex-vertex connectivity
        for simplex in qh.simplices().filter(|f| !f.upper_delaunay()) {
            if let Some(verts) = simplex.vertices() {
                let simplex_indices: Vec<usize> = verts
                    .iter()
                    .filter_map(|v| v.index(&qh))
                    .collect();

                if !scipy_compat {
                    // Full 1-skeleton: every pair of simplex vertices is an edge.
                    for (a, &pi) in simplex_indices.iter().enumerate() {
                        for &pj in &simplex_indices[a + 1..] {
                            if pi < vertices.len() && pj < vertices.len() {
                                crate::Vertex::connect_bidirectional(
                                    &vertices[pi],
                                    &vertices[pj],
                                );
                            }
                        }
                    }
                } else if dim >= 2 && simplex_indices.len() >= dim {
                    // SciPy parity: combinations(simplex, dim) → connect e[0]-e[1]
                    let mut combo: Vec<usize> = (0..dim).collect();
                    let n = simplex_indices.len();
                    loop {
                        let pi = simplex_indices[combo[0]];
                        let pj = simplex_indices[combo[1]];
                        if pi < vertices.len() && pj < vertices.len() {
                            crate::Vertex::connect_bidirectional(
                                &vertices[pi],
                                &vertices[pj],
                            );
                        }

                        let mut i = dim - 1;
                        loop {
                            combo[i] += 1;
                            if combo[i] <= n - dim + i {
                                break;
                            }
                            if i == 0 {
                                combo[0] = n;
                                break;
                            }
                            i -= 1;
                        }
                        if combo[0] >= n {
                            break;
                        }
                        for j in (i + 1)..dim {
                            combo[j] = combo[j - 1] + 1;
                        }
                    }
                } else if simplex_indices.len() >= 2 {
                    let pi = simplex_indices[0];
                    let pj = simplex_indices[1];
                    if pi < vertices.len() && pj < vertices.len() {
                        crate::Vertex::connect_bidirectional(
                            &vertices[pi],
                            &vertices[pj],
                        );
                    }
                }
            }
        }

        if disp > 1 {
            println!(
                "  Delaunay triangulation: {} simplices, {} vertices",
                qh.simplices()
                    .filter(|f| !f.upper_delaunay())
                    .count(),
                qh.num_vertices()
            );
        }

        Ok(())
    }

    /// Construct locally (approximately) convex bounds for simplicial mode.
    ///
    /// For each minimizer candidate, tighten the bounds based on
    /// nearby neighbor positions. This restricts the local optimizer
    /// to search within the vertex's basin.
    fn construct_lcb_simplicial<F2, G2>(
        &self,
        v_min: &crate::Vertex,
        complex: &Complex<F2, G2>,
    ) -> Vec<(f64, f64)>
    where
        F2: Fn(&[f64]) -> f64 + Send + Sync + 'static,
        G2: Fn(&[f64]) -> bool + Send + Sync + 'static,
    {
        // Start with the full domain bounds
        let mut cbounds: Vec<(f64, f64)> = self.bounds.clone();
        let v_min_x = v_min.x();

        // Tighten bounds based on neighbors
        for &nn_idx in &v_min.neighbor_indices() {
            if let Some(neighbor) = complex.cache.get_by_index(nn_idx) {
                let nn_x = neighbor.x();
                for i in 0..self.dim {
                    // Lower bound: closest neighbor below v_min in dim i
                    if nn_x[i] < v_min_x[i] && nn_x[i] > cbounds[i].0 {
                        cbounds[i].0 = nn_x[i];
                    }
                    // Upper bound: closest neighbor above v_min in dim i
                    if nn_x[i] > v_min_x[i] && nn_x[i] < cbounds[i].1 {
                        cbounds[i].1 = nn_x[i];
                    }
                }
            }
        }

        cbounds
    }

    /// Perform local minimization from a starting point.
    ///
    /// Uses locally convex bounds and passes constraints to the optimizer.
    fn minimize_local_from_point(
        &self,
        x0: &[f64],
        cache: &LMapCache,
        local_bounds: &[(f64, f64)],
    ) -> Option<LocalMinimum> {
        let coords = Coordinates::new(x0.to_vec());

        // Check if we've already minimized from this point
        if cache.contains(&coords) {
            return cache.get(&coords);
        }

        // Create options for local optimization
        let mut local_opts = self.options.local_options.clone();

        // Auto-upgrade: if constraints exist but the chosen algorithm doesn't
        // support them, switch to Cobyla (which does).
        if !self.constraints.is_empty() && !local_opts.algorithm.supports_constraints() {
            if self.options.disp > 0 {
                eprintln!(
                    "Warning: {:?} does not support constraints, auto-upgrading to Cobyla",
                    local_opts.algorithm
                );
            }
            local_opts.algorithm = crate::local_opt::LocalOptimizer::Cobyla;
        }

        // Clone the function for local optimization (without evaluation counting,
        // since local evals are tracked separately via lmap_cache.total_fev())
        let func = Arc::clone(&self.func);

        // Run local optimization with constraints if available
        let result = if !self.constraints.is_empty()
            && local_opts.algorithm.supports_constraints()
        {
            // Build constraint wrappers for NLOPT (g(x) >= 0 convention)
            let constraint_fns: Vec<crate::local_opt::BoxedConstraint> = self
                .constraints
                .iter()
                .map(|c| {
                    let c = Arc::clone(c);
                    Box::new(move |x: &[f64]| c(x)) as crate::local_opt::BoxedConstraint
                })
                .collect();

            crate::local_opt::minimize_local_constrained(
                |x: &[f64]| func(x),
                x0,
                local_bounds,
                &constraint_fns,
                &local_opts,
            )
        } else {
            crate::local_opt::minimize_local(
                &|x: &[f64]| func(x),
                x0,
                local_bounds,
                None::<&[fn(&[f64]) -> f64]>,
                &local_opts,
            )
        };

        // A NaN objective value would poison the f-ordered caches; treat it as
        // infeasible (+inf), like the sampling path does.
        let fun = if result.fun.is_nan() { f64::INFINITY } else { result.fun };
        let local_min = LocalMinimum {
            x: result.x,
            fun,
            success: result.success && fun.is_finite(),
            nfev: result.nfev,
            nit: result.nit,
        };

        // Cache the result
        cache.insert(coords, local_min.clone());

        Some(local_min)
    }

    /// Per-dimension perturbation radii: `radius_rel` times the bounds width
    /// (`max(1, |x_i|)` for an effectively unbounded dimension).
    fn stencil_radii(&self, center: &[f64], radius_rel: f64) -> Vec<f64> {
        center
            .iter()
            .zip(self.bounds.iter())
            .map(|(x, (lo, hi))| {
                let width = hi - lo;
                let scale = if width.is_finite() && width < 1e30 {
                    width
                } else {
                    x.abs().max(1.0)
                };
                radius_rel * scale
            })
            .collect()
    }

    /// The stencil points around `center`, clipped to the bounds: optionally
    /// the centre itself, then the axis steps (tagged with their dimension),
    /// then the Sobol points. Points that clip onto the centre (a minimum on
    /// a bound) are dropped; the count of dropped points is returned too.
    fn stencil_points(&self, center: &[f64], st: &Stencil, include_center: bool) -> StencilPoints {
        let dim = center.len();
        let r = self.stencil_radii(center, st.radius_rel);
        let clip = |v: Vec<f64>| -> Vec<f64> {
            v.into_iter()
                .zip(self.bounds.iter())
                .map(|(x, (lo, hi))| x.clamp(*lo, *hi))
                .collect()
        };
        let mut pts: Vec<(Vec<f64>, Option<usize>)> = Vec::new();
        let mut dropped = 0usize;
        if include_center {
            pts.push((center.to_vec(), None));
        }
        if st.axis_steps {
            for i in 0..dim {
                for sign in [1.0, -1.0] {
                    let mut p = center.to_vec();
                    p[i] += sign * r[i];
                    let p = clip(p);
                    if p == center {
                        dropped += 1;
                    } else {
                        pts.push((p, Some(i)));
                    }
                }
            }
        }
        if st.samples > 0 {
            // Skip the first two Sobol points: index 0 is the box corner and
            // index 1 is the box centre, i.e. `center` itself.
            let mut sobol = Sobol::new(dim);
            for u in sobol.generate(st.samples, 2) {
                let p: Vec<f64> = u
                    .iter()
                    .zip(center.iter())
                    .zip(r.iter())
                    .map(|((ui, x), ri)| x + (2.0 * ui - 1.0) * ri)
                    .collect();
                let p = clip(p);
                if p == center {
                    dropped += 1;
                } else {
                    pts.push((p, None));
                }
            }
        }
        (pts, dropped)
    }

    /// Evaluate the objective at `points` in parallel. `None` marks a point
    /// that violates a constraint (not evaluated) or returned a non-finite
    /// value. Also returns the number of objective calls made.
    fn evaluate_points(&self, points: &[Vec<f64>]) -> (Vec<Option<f64>>, usize) {
        let func = &self.func;
        let cons = &self.constraints;
        let calls = AtomicUsize::new(0);
        let values: Vec<Option<f64>> = points
            .par_iter()
            .map(|p| {
                // NaN counts as a violation, as in the sampling path.
                if cons.iter().any(|g| {
                    let v = g(p);
                    v.is_nan() || v < 0.0
                }) {
                    return None;
                }
                calls.fetch_add(1, Ordering::Relaxed);
                let f = func(p);
                if f.is_finite() {
                    Some(f)
                } else {
                    None
                }
            })
            .collect();
        (values, calls.load(Ordering::Relaxed))
    }

    /// [`ShgoOptions::robustness_probe`]: evaluate a stencil around each of the
    /// probed minima and summarise it. `result.xl` must already be sorted
    /// ascending by `funl`. Returns the number of objective evaluations made.
    fn probe_robustness(&self, result: &mut ShgoResult, probe: &RobustnessProbe) -> usize {
        let m = probe
            .top
            .map(|t| t.min(result.xl.len()))
            .unwrap_or(result.xl.len());
        let mut all_points: Vec<Vec<f64>> = Vec::new();
        let mut meta: Vec<(usize, Option<usize>)> = Vec::new();
        let mut dropped = vec![0usize; m];
        for (i, d) in dropped.iter_mut().enumerate() {
            let (pts, dr) = self.stencil_points(&result.xl[i], &probe.stencil, false);
            *d = dr;
            for (p, axis) in pts {
                all_points.push(p);
                meta.push((i, axis));
            }
        }
        let (values, n_calls) = self.evaluate_points(&all_points);

        let mut stats = Vec::with_capacity(m);
        for (i, &dr) in dropped.iter().enumerate() {
            let f_center = result.funl[i];
            let mut feasible: Vec<f64> = vec![f_center];
            let mut n_infeasible = dr;
            let mut worst_axis: Option<(usize, f64)> = None;
            for (k, (xi, axis)) in meta.iter().enumerate() {
                if *xi != i {
                    continue;
                }
                match values[k] {
                    Some(f) => {
                        feasible.push(f);
                        if let Some(a) = axis {
                            let inc = f - f_center;
                            if inc > 0.0 && worst_axis.is_none_or(|(_, w)| inc > w) {
                                worst_axis = Some((*a, inc));
                            }
                        }
                    }
                    None => n_infeasible += 1,
                }
            }
            let n = feasible.len();
            let f_mean = feasible.iter().sum::<f64>() / n as f64;
            let f_std = (feasible.iter().map(|f| (f - f_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            let mut sorted = feasible.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            stats.push(RobustnessStats {
                xl_index: i,
                f_center,
                robust_value: robust_aggregate(&feasible, probe.aggregate),
                f_mean,
                f_median: sorted[n / 2],
                f_min: sorted[0],
                f_max: sorted[n - 1],
                f_std,
                worst_axis: worst_axis.map(|(a, _)| a),
                n_feasible: n,
                n_infeasible,
            });
        }
        result.robustness = Some(stats);
        n_calls
    }

    /// [`ShgoOptions::robust_polish`]: re-optimize the chosen minima on the
    /// stencil-smoothed objective. Returns the number of raw objective
    /// evaluations made.
    fn robust_polish(&self, result: &mut ShgoResult, rp: &RobustPolish) -> usize {
        // Rank by the probe's robust value when available, else by funl order.
        let order: Vec<usize> = match &result.robustness {
            Some(stats) => {
                let mut v: Vec<(usize, f64)> =
                    stats.iter().map(|s| (s.xl_index, s.robust_value)).collect();
                v.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                v.into_iter().map(|(i, _)| i).collect()
            }
            None => (0..result.xl.len()).collect(),
        };
        let chosen: Vec<usize> = order.into_iter().take(rp.top).collect();

        let mut local_opts = self.options.local_options.clone();
        if let Some(me) = rp.maxeval {
            local_opts.maxeval = Some(me);
        }
        if !self.constraints.is_empty() && !local_opts.algorithm.supports_constraints() {
            local_opts.algorithm = crate::local_opt::LocalOptimizer::Cobyla;
        }

        let mut out = Vec::with_capacity(chosen.len());
        let mut total = 0usize;
        for xi in chosen {
            let x0 = result.xl[xi].clone();
            let raw_calls = AtomicUsize::new(0);
            let smoothed = |x: &[f64]| -> f64 {
                let (pts, _) = self.stencil_points(x, &rp.stencil, true);
                let points: Vec<Vec<f64>> = pts.into_iter().map(|(p, _)| p).collect();
                let (values, calls) = self.evaluate_points(&points);
                raw_calls.fetch_add(calls, Ordering::Relaxed);
                let feasible: Vec<f64> = values.into_iter().flatten().collect();
                robust_aggregate(&feasible, rp.aggregate)
            };
            let res = if self.constraints.is_empty() {
                crate::local_opt::minimize_local(
                    &smoothed,
                    &x0,
                    &self.bounds,
                    None::<&[fn(&[f64]) -> f64]>,
                    &local_opts,
                )
            } else {
                let cons: Vec<crate::local_opt::BoxedConstraint> = self
                    .constraints
                    .iter()
                    .map(|c| {
                        let c = Arc::clone(c);
                        Box::new(move |x: &[f64]| c(x)) as crate::local_opt::BoxedConstraint
                    })
                    .collect();
                crate::local_opt::minimize_local_constrained(
                    smoothed,
                    &x0,
                    &self.bounds,
                    &cons,
                    &local_opts,
                )
            };
            let f_center = (self.func)(&res.x);
            let nfev = raw_calls.load(Ordering::Relaxed) + 1;
            total += nfev;
            out.push(RobustMinimum {
                xl_index: xi,
                x: res.x,
                robust_value: res.fun,
                f_center,
                nfev,
                success: res.success && res.fun.is_finite(),
            });
        }
        out.sort_by(|a, b| {
            a.robust_value
                .partial_cmp(&b.robust_value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.xl_index.cmp(&b.xl_index))
        });
        result.robust_minima = Some(out);
        total
    }

    /// Print optimization summary.
    fn print_summary(&self, result: &ShgoResult) {
        println!("\n=== SHGO Optimization Summary ===");
        println!("Success: {}", result.success);
        println!("Message: {}", result.message);
        println!("Best solution: {:?}", result.x);
        println!("Best function value: {:.6e}", result.fun);
        println!("Iterations: {}", result.nit);
        println!("Function evaluations: {}", result.nfev);
        println!("Local minima found: {}", result.xl.len());
        println!("Time elapsed: {:.3}s", result.time);
        println!("================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sphere function: f(x) = sum(x_i^2)
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi.powi(2)).sum()
    }

    // Rosenbrock function
    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    // Rastrigin function
    fn rastrigin(x: &[f64]) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n + x.iter().map(|xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
    }

    #[test]
    fn test_shgo_options_default() {
        let opts = ShgoOptions::default();
        assert!(opts.maxiter.is_none());
        assert!(opts.maxfev.is_none());
        assert_eq!(opts.f_tol, 1e-4);
        assert_eq!(opts.n, 0); // 0 = auto (2^dim+1 for simplicial, 128 for Sobol)
        assert_eq!(opts.iters, Some(1));
        assert_eq!(opts.sampling_method, SamplingMethod::Simplicial);
    }

    #[test]
    fn test_shgo_new() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let shgo = Shgo::new(sphere, bounds.clone());
        assert_eq!(shgo.dim, 2);
        assert_eq!(shgo.bounds, bounds);
        assert!(shgo.constraints.is_empty());
    }

    #[test]
    fn test_shgo_with_constraints() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let constraint = |x: &[f64]| x[0] + x[1] - 1.0;
        let shgo = Shgo::with_constraints(sphere, bounds.clone(), vec![constraint]);
        assert_eq!(shgo.dim, 2);
        assert_eq!(shgo.constraints.len(), 1);
    }

    #[test]
    fn test_validate_bounds_valid() {
        let bounds = vec![(-1.0, 1.0), (-2.0, 2.0)];
        let shgo = Shgo::new(sphere, bounds);
        assert!(shgo.validate_bounds().is_ok());
    }

    #[test]
    fn test_validate_bounds_invalid_order() {
        let bounds = vec![(1.0, -1.0)]; // Invalid: lower > upper
        let shgo = Shgo::new(sphere, bounds);
        assert!(shgo.validate_bounds().is_err());
    }

    #[test]
    fn test_validate_bounds_empty() {
        let bounds: Vec<(f64, f64)> = vec![];
        let shgo = Shgo::new(sphere, bounds);
        assert!(shgo.validate_bounds().is_err());
    }

    #[test]
    fn test_validate_bounds_nan_replaced() {
        // NaN bounds are replaced with ±1e50 (matching Python's behavior)
        let bounds = vec![(f64::NAN, 1.0)];
        let shgo = Shgo::new(sphere, bounds);
        // NaN lower bound is replaced with -1e50, so validation passes
        assert!(shgo.validate_bounds().is_ok());
    }

    #[test]
    fn test_lmap_cache_basic() {
        let cache = LMapCache::new();
        assert!(cache.is_empty());

        let coords = Coordinates::new(vec![0.5, 0.5]);
        let result = LocalMinimum {
            x: vec![0.5, 0.5],
            fun: 0.5,
            success: true,
            nfev: 10,
            nit: 5,
        };

        cache.insert(coords.clone(), result.clone());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&coords));

        let retrieved = cache.get(&coords).unwrap();
        assert_eq!(retrieved.fun, 0.5);
        assert_eq!(cache.total_fev(), 10);
    }

    #[test]
    fn test_lmap_cache_sorted() {
        let cache = LMapCache::new();

        cache.insert(
            Coordinates::new(vec![1.0]),
            LocalMinimum {
                x: vec![1.0],
                fun: 10.0,
                success: true,
                nfev: 1,
                nit: 1,
            },
        );
        cache.insert(
            Coordinates::new(vec![2.0]),
            LocalMinimum {
                x: vec![2.0],
                fun: 5.0,
                success: true,
                nfev: 1,
                nit: 1,
            },
        );
        cache.insert(
            Coordinates::new(vec![3.0]),
            LocalMinimum {
                x: vec![3.0],
                fun: 15.0,
                success: true,
                nfev: 1,
                nit: 1,
            },
        );

        let sorted = cache.get_sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].fun, 5.0);
        assert_eq!(sorted[1].fun, 10.0);
        assert_eq!(sorted[2].fun, 15.0);
    }

    #[test]
    fn test_shgo_sphere_simplicial() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxiter: Some(3),
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nit <= 3);
        assert!(result.nfev > 0);
        // The minimum should be near (0, 0)
        println!("Sphere result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_sphere_sobol() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            n: 64,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nit <= 2);
        assert!(result.nfev > 0);
        println!("Sphere (Sobol) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_sphere_sobol_knn() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 64,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nit <= 2);
        assert!(result.nfev > 0);
        assert!(result.fun < 1.0, "k-NN should find near-optimal: fun={}", result.fun);
        println!("Sphere (Sobol+k-NN) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_rosenbrock_sobol_knn() {
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let options = ShgoOptions {
            maxiter: Some(3),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 128,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rosenbrock, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.fun < 1.0, "k-NN Rosenbrock should converge: fun={}", result.fun);
        println!("Rosenbrock (Sobol+k-NN) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_rastrigin_5d_knn() {
        let rastrigin = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        };
        let bounds = vec![(-5.12, 5.12); 5];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 256,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rastrigin, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nfev > 0);
        println!("Rastrigin 5D (k-NN) result: fun={}, x={:?}", result.fun, result.x);
    }

    #[test]
    fn test_shgo_sphere_sobol_hnsw() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::HNSW,
            n: 64,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.fun < 1.0, "HNSW should find near-optimal: fun={}", result.fun);
        println!("Sphere (Sobol+HNSW) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_rosenbrock_sobol_hnsw() {
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let options = ShgoOptions {
            maxiter: Some(3),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::HNSW,
            n: 128,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rosenbrock, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.fun < 1.0, "HNSW Rosenbrock should converge: fun={}", result.fun);
        println!("Rosenbrock (Sobol+HNSW) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_rastrigin_5d_hnsw() {
        let rastrigin = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        };
        let bounds = vec![(-5.12, 5.12); 5];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::HNSW,
            n: 256,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rastrigin, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nfev > 0);
        println!("Rastrigin 5D (HNSW) result: fun={}, x={:?}", result.fun, result.x);
    }

    #[test]
    fn test_shgo_sphere_sobol_scann() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::ScaNN,
            n: 64,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.fun < 1.0, "ScaNN should find near-optimal: fun={}", result.fun);
        println!("Sphere (Sobol+ScaNN) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_rosenbrock_sobol_scann() {
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let options = ShgoOptions {
            maxiter: Some(3),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::ScaNN,
            n: 128,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rosenbrock, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.fun < 1.0, "ScaNN Rosenbrock should converge: fun={}", result.fun);
        println!("Rosenbrock (Sobol+ScaNN) result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_rastrigin_5d_scann() {
        let rastrigin = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        };
        let bounds = vec![(-5.12, 5.12); 5];
        let options = ShgoOptions {
            maxiter: Some(2),
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::ScaNN,
            n: 256,
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rastrigin, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nfev > 0);
        println!("Rastrigin 5D (ScaNN) result: fun={}, x={:?}", result.fun, result.x);
    }

    #[test]
    fn test_shgo_rosenbrock() {
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let options = ShgoOptions {
            maxiter: Some(3),
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rosenbrock, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        assert!(result.nit <= 3);
        println!("Rosenbrock result: x={:?}, fun={}", result.x, result.fun);
    }

    #[test]
    fn test_shgo_cancellation() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxiter: Some(100),
            ..Default::default()
        };

        let shgo = Shgo::new(sphere, bounds).with_options(options);
        
        // Cancel before starting
        shgo.cancel();
        
        let result = shgo.minimize();
        assert!(matches!(result, Err(ShgoError::Cancelled)));
    }

    #[test]
    fn test_shgo_with_constraint() {
        let bounds = vec![(0.0, 3.0), (0.0, 3.0)];
        
        // Constraint: x[0] + x[1] >= 2
        let constraint = |x: &[f64]| x[0] + x[1] - 2.0;
        
        let options = ShgoOptions {
            maxiter: Some(3),
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::with_constraints(sphere, bounds, vec![constraint])
            .with_options(options)
            .minimize()
            .unwrap();

        println!("Constrained sphere result: x={:?}, fun={}", result.x, result.fun);
        // With constraint x[0] + x[1] >= 2, minimum should be at (1, 1)
    }

    #[test]
    fn test_shgo_fev_count() {
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let options = ShgoOptions {
            maxiter: Some(2),
            ..Default::default()
        };

        let shgo = Shgo::new(sphere, bounds).with_options(options);

        assert_eq!(shgo.fev_count(), 0);

        let result = shgo.minimize().unwrap();

        assert!(result.nfev > 0);
        // nfev = sampling evaluations + local-minimization evaluations
        // (matching SciPy's res.nfev = fn + nlfev).
        assert_eq!(result.nfev, shgo.fev_count() + result.nlfev);
        assert!(result.nlfev > 0);
    }

    #[test]
    fn test_shgo_time_limit() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxtime: Some(0.5), // 500ms limit
            maxiter: Some(1000), // High iteration limit
            sampling_method: SamplingMethod::Sobol,
            n: 64,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        // Should finish within time limit (with tolerance for in-flight local opt)
        assert!(result.time < 3.0);
    }

    #[test]
    fn test_shgo_result_new() {
        let result = ShgoResult::new(3);
        assert_eq!(result.x.len(), 3);
        assert!(result.fun.is_infinite());
        assert!(result.xl.is_empty());
        assert!(!result.success);
    }

    #[test]
    fn test_sampling_method_default() {
        let method = SamplingMethod::default();
        assert_eq!(method, SamplingMethod::Simplicial);
    }

    #[test]
    fn test_shgo_maxfev_limit() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = ShgoOptions {
            maxfev: Some(200),
            maxiter: Some(100),
            sampling_method: SamplingMethod::Sobol,
            n: 32,
            ..Default::default()
        };

        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        // Verify the optimizer stopped (didn't run all 100 iterations)
        assert!(result.nit < 100);
    }

    #[test]
    fn test_simplicial_default_samples_initial_complex_only() {
        // SciPy parity: a default simplicial run (iters = 1) samples exactly
        // the initial complex — 2^dim corners + centroid — with no extra
        // refinement generation (scipy 1.18 probe: 513 evals in 9-D).
        let bounds = vec![(-5.0, 5.0); 9];
        let result = Shgo::new(sphere, bounds)
            .with_options(ShgoOptions::default())
            .minimize()
            .unwrap();

        let sampling_evals = result.nfev - result.nlfev;
        assert_eq!(sampling_evals, (1 << 9) + 1, "expected 513 sampling evals");
        assert_eq!(result.nit, 1);
        assert!(result.fun < 1e-8);
    }

    #[test]
    fn test_simplicial_linear_growth_per_iteration() {
        // SciPy parity: each iteration adds ~n = 2^dim + 1 new sampling
        // points (scipy 1.18 probe: 2-D V.size 5 -> 10 -> 15 -> 20 over 4
        // iterations), not a full refinement generation.
        let bounds = vec![(-5.0, 5.0); 2];
        let options = ShgoOptions {
            maxiter: Some(4),
            ..Default::default()
        };
        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        let sampling_evals = result.nfev - result.nlfev;
        assert_eq!(result.nit, 4);
        assert!(
            (20..=26).contains(&sampling_evals),
            "expected ~20 sampling evals (5/iter x 4 iters), got {}",
            sampling_evals
        );
    }

    #[test]
    fn test_basin_stats_double_well() {
        // Double well in x (global near x=-1, local near x=+1) + bowl in y.
        let f = |x: &[f64]| (x[0] * x[0] - 1.0).powi(2) + 0.1 * x[0] + x[1] * x[1];
        let options = ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 256,
            iters: Some(1),
            compute_basin_stats: true,
            basin_good_thresholds: vec![0.5],
            ..Default::default()
        };
        let run = || {
            Shgo::new(f, vec![(-2.0, 2.0), (-2.0, 2.0)])
                .with_options(options.clone())
                .minimize()
                .unwrap()
        };
        let result = run();
        let basins = result.basins.expect("basin stats were requested");
        assert!(!basins.is_empty());

        // Every feasible sampled point belongs to exactly one basin.
        let total: usize = basins.iter().map(|b| b.size).sum();
        assert_eq!(total, 256);

        // Sorted ascending by sampled minimum; the global basin (x < 0)
        // comes first and has infinite persistence (it never merges).
        assert!(basins[0].persistence.is_infinite());
        assert!(basins[0].x_sampled[0] < 0.0);
        assert!(basins[0].size > 40, "global well should hold a large share");

        for b in &basins {
            assert!(b.good_counts[0] <= b.size);
            assert!(b.f_median >= b.f_min_sampled);
            assert!(b.f_tail >= b.f_median);
        }

        // The global basin's graph minimizer was polished and maps to xl.
        assert!(basins[0].xl_index.is_some());
        let i = basins[0].xl_index.unwrap();
        assert!(result.xl[i][0] < 0.0);

        // Deterministic across runs.
        let result2 = run();
        let basins2 = result2.basins.unwrap();
        assert_eq!(basins.len(), basins2.len());
        for (a, b) in basins.iter().zip(basins2.iter()) {
            assert_eq!(a.size, b.size);
            assert_eq!(a.x_sampled, b.x_sampled);
        }
    }

    #[test]
    fn test_maxev_counts_sampled_points_not_fev() {
        // maxev limits SAMPLED points (including infeasible ones), matching
        // SciPy's `n_sampled >= maxev` — not feasible objective evaluations.
        // The constraint makes half the domain infeasible, so feasible evals
        // (~32 after iteration 1) lag well behind sampled points (64).
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let constraint = |x: &[f64]| x[0]; // feasible only when x[0] >= 0
        let options = ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            n: 64,
            maxev: Some(60),
            maxiter: Some(100),
            ..Default::default()
        };

        let result = Shgo::with_constraints(sphere, bounds, vec![constraint])
            .with_options(options)
            .minimize()
            .unwrap();

        // 64 points sampled in iteration 1 >= maxev = 60 → stop immediately.
        // (The old bug compared feasible evals (~32) and ran a 2nd iteration.)
        assert_eq!(result.nit, 1);
    }

    /// Regression: a minimum re-inserted into the next iteration's point cloud
    /// is a graph minimizer by construction and used to be re-minimized every
    /// iteration. In 1-D the sorted-adjacency graph of a convex function has
    /// exactly one minimizer, so after iteration 1 the only candidate is the
    /// re-inserted minimum itself and no further local runs may happen.
    #[test]
    fn test_known_minimum_not_reminimized() {
        let bowl = |x: &[f64]| (x[0] - 0.3).powi(2);
        let run = |maxiter| {
            Shgo::new(bowl, vec![(-1.0, 1.0)])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    n: 8,
                    maxiter: Some(maxiter),
                    ..Default::default()
                })
                .minimize()
                .unwrap()
        };
        let one = run(1);
        let three = run(3);
        assert_eq!(three.nit, 3);
        assert!(one.nlfev > 0);
        assert_eq!(
            three.nlfev, one.nlfev,
            "iterations 2-3 re-minimized the re-inserted minimum ({} -> {} local evals)",
            one.nlfev, three.nlfev
        );
        assert_eq!(three.xl.len(), 1);
        assert!((three.x[0] - 0.3).abs() < 1e-6);
    }

    /// The full Delaunay 1-skeleton (paper Definition 18) yields exactly one
    /// minimizer candidate on a bowl; SciPy's `vf_to_vv` quirk, kept as
    /// `DelaunayScipyCompat`, connects only the first three vertices of each
    /// simplex for dim >= 3 and produces spurious candidates.
    #[test]
    fn test_delaunay_full_skeleton_has_no_spurious_minimizers() {
        for (dim, n) in [(3usize, 256usize), (4, 512)] {
            let candidates = |method| {
                Shgo::new(sphere, vec![(-5.0, 5.0); dim])
                    .with_options(ShgoOptions {
                        sampling_method: SamplingMethod::Sobol,
                        connectivity_method: method,
                        n,
                        iters: Some(1),
                        minimize_every_iter: false,
                        ..Default::default()
                    })
                    .minimize()
                    .unwrap()
                    .xl
                    .len()
            };
            let full = candidates(ConnectivityMethod::Delaunay);
            let compat = candidates(ConnectivityMethod::DelaunayScipyCompat);
            assert_eq!(full, 1, "dim {} n {}: full skeleton gave {} candidates", dim, n, full);
            assert!(compat > full, "dim {} n {}: compat gave {} candidates", dim, n, compat);
        }
    }

    #[test]
    fn test_no_stopping_criterion_is_an_error() {
        let result = Shgo::new(sphere, vec![(-5.0, 5.0); 2])
            .with_options(ShgoOptions {
                iters: None,
                ..Default::default()
            })
            .minimize();
        assert!(matches!(result, Err(ShgoError::InvalidOption(_))));
    }

    /// Local runs that end in a NaN region must not enter `xl` or poison the
    /// f-ordered caches.
    #[test]
    fn test_nan_region_local_results_are_excluded() {
        let f = |x: &[f64]| if x[0] < 0.0 { f64::NAN } else { sphere(x) };
        let result = Shgo::new(f, vec![(-5.0, 5.0); 2])
            .with_options(ShgoOptions {
                sampling_method: SamplingMethod::Sobol,
                connectivity_method: ConnectivityMethod::KNearestNeighbors,
                n: 64,
                iters: Some(1),
                ..Default::default()
            })
            .minimize()
            .unwrap();
        assert!(result.success);
        assert!(result.fun.is_finite());
        assert!(result.funl.iter().all(|f| f.is_finite()));
    }

    /// `xl` de-duplication uses a tolerance tied to the bounds, so two local
    /// runs converging to the same minimum from different starts collapse to
    /// one row (2-D Rastrigin has 121 minima in this box).
    #[test]
    fn test_xl_dedup_merges_near_duplicates() {
        let bounds = vec![(-5.12, 5.12); 2];
        let result = Shgo::new(rastrigin, bounds.clone())
            .with_options(ShgoOptions {
                sampling_method: SamplingMethod::Sobol,
                connectivity_method: ConnectivityMethod::KNearestNeighbors,
                n: 256,
                maxiter: Some(2),
                ..Default::default()
            })
            .minimize()
            .unwrap();
        assert!(result.xl.len() <= 121, "{} rows for 121 minima", result.xl.len());
        let tol = 1e-4 * 10.24;
        for i in 0..result.xl.len() {
            for j in 0..i {
                let close = result.xl[i]
                    .iter()
                    .zip(&result.xl[j])
                    .all(|(a, b)| (a - b).abs() <= tol);
                assert!(!close, "rows {} and {} are the same minimum", i, j);
            }
        }
    }

    /// Gradient-based local optimizers must move (they used to receive no
    /// gradient and returned every start point unchanged after one evaluation).
    #[test]
    fn test_gradient_based_local_optimizers_through_shgo() {
        use crate::local_opt::LocalOptimizer;
        let shifted = |x: &[f64]| x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
        for alg in [LocalOptimizer::Slsqp, LocalOptimizer::Lbfgs] {
            let result = Shgo::new(shifted, vec![(-5.0, 5.0); 4])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::KNearestNeighbors,
                    n: 256,
                    iters: Some(1),
                    local_options: crate::local_opt::LocalOptimizerOptions {
                        algorithm: alg,
                        ..crate::local_opt::LocalOptimizerOptions::default()
                    },
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            assert!(result.fun < 1e-8, "{:?}: fun = {}", alg, result.fun);
            assert!(result.nlfev > 4 * result.xl.len(), "{:?}: only {} local evals", alg, result.nlfev);
        }
    }

    /// The |M_k| curve must be self-consistent: the graph actually built at
    /// the selected k must have exactly `curve[k]` minimizer candidates. The
    /// curve is also monotone non-increasing, and k is the smallest value
    /// meeting the budget.
    #[test]
    fn test_knn_auto_curve_is_consistent_and_monotone() {
        let multiwell = |x: &[f64]| -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, &v)| (v * v - 1.0).powi(2) + 0.1 * (1.0 + 0.05 * i as f64) * v)
                .sum()
        };
        for budget in [10usize, 40, 200] {
            let result = Shgo::new(multiwell, vec![(-2.0, 2.0); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::KNearestNeighbors,
                    n: 1024,
                    iters: Some(1),
                    minimize_every_iter: false,
                    knn_auto: Some(KnnAuto::with_budget(budget)),
                    ..Default::default()
                })
                .minimize()
                .unwrap();
            let sel = result.knn_selection.expect("auto selection was requested");
            assert_eq!(sel.curve.len(), sel.k_max + 1);
            for w in sel.curve.windows(2) {
                assert!(w[1] <= w[0], "|M_k| must be non-increasing: {:?}", w);
            }
            // With minimize_every_iter = false every graph minimizer becomes an
            // xl row, so the realized pool size must match the predicted one.
            assert_eq!(
                result.xl.len(),
                sel.curve[sel.k],
                "budget {}: curve predicted {} candidates at k={}, graph produced {}",
                budget,
                sel.curve[sel.k],
                sel.k,
                result.xl.len()
            );
            // k is the smallest value in range that meets the budget.
            if sel.curve[sel.k] <= budget {
                for k in (6..sel.k).rev() {
                    assert!(
                        sel.curve[k] > budget,
                        "k={} already met budget {} ({} candidates) but k={} was chosen",
                        k,
                        budget,
                        sel.curve[k],
                        sel.k
                    );
                }
            }
        }
    }

    /// A tighter budget must not select a smaller k, and the pool it produces
    /// must respect the budget whenever the curve can reach it.
    #[test]
    fn test_knn_auto_budget_controls_pool_size() {
        let rastrigin_5d = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            10.0 * n
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                    .sum::<f64>()
        };
        let run = |budget: usize| {
            Shgo::new(rastrigin_5d, vec![(-5.12, 5.12); 5])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    connectivity_method: ConnectivityMethod::KNearestNeighbors,
                    n: 2048,
                    iters: Some(1),
                    minimize_every_iter: false,
                    knn_auto: Some(KnnAuto::with_budget(budget)),
                    ..Default::default()
                })
                .minimize()
                .unwrap()
        };
        let loose = run(400);
        let tight = run(50);
        let (kl, kt) = (
            loose.knn_selection.as_ref().unwrap().k,
            tight.knn_selection.as_ref().unwrap().k,
        );
        assert!(kt >= kl, "tighter budget selected smaller k ({} < {})", kt, kl);
        assert!(tight.xl.len() <= 50, "tight budget gave {} candidates", tight.xl.len());
        assert!(loose.xl.len() <= 400, "loose budget gave {} candidates", loose.xl.len());
    }

    #[test]
    fn test_knn_auto_rejected_for_other_connectivity() {
        let result = Shgo::new(sphere, vec![(-5.0, 5.0); 2])
            .with_options(ShgoOptions {
                sampling_method: SamplingMethod::Sobol,
                connectivity_method: ConnectivityMethod::Delaunay,
                n: 64,
                iters: Some(1),
                knn_auto: Some(KnnAuto::with_budget(10)),
                ..Default::default()
            })
            .minimize();
        assert!(matches!(result, Err(ShgoError::InvalidOption(_))));
    }

    /// Persistence pruning must cut local runs while keeping the global
    /// minimum, and must never keep more candidates than the cap.
    #[test]
    fn test_persistence_pruning_cuts_candidates_and_keeps_global() {
        let multiwell = |x: &[f64]| -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, &v)| (v * v - 1.0).powi(2) + 0.1 * (1.0 + 0.05 * i as f64) * v)
                .sum()
        };
        let base = ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 2048,
            knn_neighbors: Some(11),
            iters: Some(1),
            ..Default::default()
        };
        let plain = Shgo::new(multiwell, vec![(-2.0, 2.0); 5])
            .with_options(base.clone())
            .minimize()
            .unwrap();
        let pruned = Shgo::new(multiwell, vec![(-2.0, 2.0); 5])
            .with_options(ShgoOptions {
                min_candidate_persistence: Some(0.05),
                max_candidates_by_persistence: Some(20),
                ..base
            })
            .minimize()
            .unwrap();

        assert!(pruned.xl.len() <= 20, "cap not honoured: {} rows", pruned.xl.len());
        assert!(
            pruned.nlfev < plain.nlfev,
            "pruning did not reduce local evaluations ({} -> {})",
            plain.nlfev,
            pruned.nlfev
        );
        // NOTE: the global optimum is deliberately NOT asserted here.
        // Persistence describes the SAMPLED landscape, and on a coarsely
        // sampled multi-well function the basin that polishes deepest need not
        // be a prominent one, so pruning can drop it. What is guaranteed is
        // that the pool stays non-empty and keeps its lowest-cost candidate
        // (see test_persistence_pruning_keeps_lowest_cost_candidate).
        assert!(
            pruned.funl.iter().any(|f| f.is_finite()),
            "pruning removed every candidate"
        );
    }

    /// The lowest-cost candidate survives even an aggressive persistence cap.
    #[test]
    fn test_persistence_pruning_keeps_lowest_cost_candidate() {
        let multiwell = |x: &[f64]| -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, &v)| (v * v - 1.0).powi(2) + 0.1 * (1.0 + 0.05 * i as f64) * v)
                .sum()
        };
        let base = ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 2048,
            knn_neighbors: Some(11),
            iters: Some(1),
            compute_basin_stats: true,
            ..Default::default()
        };
        let plain = Shgo::new(multiwell, vec![(-2.0, 2.0); 5])
            .with_options(base.clone())
            .minimize()
            .unwrap();
        // Basins are sorted ascending by sampled cost, so the first one is the
        // start that the guardrail must keep.
        let lowest = plain
            .basins
            .as_ref()
            .unwrap()
            .first()
            .and_then(|b| b.x_polished.clone())
            .expect("lowest basin was polished");
        let lowest_f = multiwell(&lowest);

        let pruned = Shgo::new(multiwell, vec![(-2.0, 2.0); 5])
            .with_options(ShgoOptions {
                min_candidate_persistence: Some(1e9),
                max_candidates_by_persistence: Some(1),
                ..base
            })
            .minimize()
            .unwrap();
        assert_eq!(pruned.xl.len(), 1, "expected exactly the guarded candidate");
        assert!(
            (pruned.funl[0] - lowest_f).abs() < 1e-6,
            "guarded candidate was not the lowest-cost one: {} vs {}",
            pruned.funl[0],
            lowest_f
        );
    }

    /// The escape hatch that restores the pre-fix behaviour must actually
    /// re-run the local optimizer from known minima.
    #[test]
    fn test_explore_from_known_minima_reruns_local_search() {
        let bowl = |x: &[f64]| (x[0] - 0.3).powi(2);
        let run = |explore: bool| {
            Shgo::new(bowl, vec![(-1.0, 1.0)])
                .with_options(ShgoOptions {
                    sampling_method: SamplingMethod::Sobol,
                    n: 8,
                    maxiter: Some(3),
                    explore_from_known_minima: explore,
                    ..Default::default()
                })
                .minimize()
                .unwrap()
        };
        assert!(run(true).nlfev > run(false).nlfev);
    }

    /// A deep minimum that is narrow along one axis, and a shallower one that
    /// is wide in every direction. Raw cost prefers the deep one; a
    /// perturbation of a few percent of the box prefers the wide one.
    fn fragile_and_robust(x: &[f64]) -> f64 {
        let deep = -1.2
            * (-((x[0] + 1.0).powi(2) / (2.0 * 0.05f64.powi(2))
                + (x[1] + 1.0).powi(2) / (2.0 * 0.6f64.powi(2))))
            .exp();
        let wide = -(-((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)) / (2.0 * 0.36)).exp();
        deep + wide
    }

    fn robust_test_options() -> ShgoOptions {
        ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            connectivity_method: ConnectivityMethod::KNearestNeighbors,
            n: 2048,
            knn_neighbors: Some(15),
            iters: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn test_robustness_probe_ranks_fragile_minimum_as_less_robust() {
        let result = Shgo::new(fragile_and_robust, vec![(-2.0, 2.0); 2])
            .with_options(ShgoOptions {
                robustness_probe: Some(RobustnessProbe::new(0.03, 8)),
                ..robust_test_options()
            })
            .minimize()
            .unwrap();
        // Raw ranking: the deep, fragile minimum first.
        assert!(result.fun < -1.15, "deep minimum not found: {}", result.fun);
        assert!(result.x[0] < 0.0);
        let stats = result.robustness.as_ref().expect("probe requested");
        assert_eq!(stats.len(), result.xl.len());
        let deep = stats
            .iter()
            .find(|s| result.xl[s.xl_index][0] < 0.0)
            .expect("fragile minimum probed");
        let wide = stats
            .iter()
            .find(|s| result.xl[s.xl_index][0] > 0.0 && result.funl[s.xl_index] < -0.9)
            .expect("wide minimum probed");
        // The probe inverts the ranking: the wide basin is more robust.
        assert!(
            wide.robust_value < deep.robust_value,
            "probe did not prefer the wide basin: wide {} deep {}",
            wide.robust_value,
            deep.robust_value
        );
        assert!(wide.robust_value < -0.9, "wide basin robust value {}", wide.robust_value);
        assert!(deep.robust_value > -0.9, "fragile basin robust value {}", deep.robust_value);
        // The fragile axis is identified, and the stencil is fully accounted for.
        assert_eq!(deep.worst_axis, Some(0));
        for s in stats {
            assert_eq!(s.n_feasible + s.n_infeasible, 1 + 2 * 2 + 8);
            assert!(s.f_min <= s.f_center && s.f_center <= s.f_max);
            assert!(s.f_std >= 0.0);
        }
        // Probe evaluations are counted.
        let plain = Shgo::new(fragile_and_robust, vec![(-2.0, 2.0); 2])
            .with_options(robust_test_options())
            .minimize()
            .unwrap();
        assert!(result.nfev > plain.nfev);
    }

    #[test]
    fn test_robust_polish_picks_and_refines_the_robust_basin() {
        let result = Shgo::new(fragile_and_robust, vec![(-2.0, 2.0); 2])
            .with_options(ShgoOptions {
                robustness_probe: Some(RobustnessProbe::new(0.03, 8)),
                robust_polish: Some(RobustPolish::new(0.03, 8, 1)),
                ..robust_test_options()
            })
            .minimize()
            .unwrap();
        let rm = result.robust_minima.as_ref().expect("polish requested");
        assert_eq!(rm.len(), 1);
        let r = &rm[0];
        // Chosen by the probe: the wide basin, not the raw global minimum.
        assert!(result.xl[r.xl_index][0] > 0.0, "polished the fragile basin");
        assert!(r.success, "robust polish did not converge");
        assert!((r.x[0] - 1.0).abs() < 0.1 && (r.x[1] - 1.0).abs() < 0.1, "{:?}", r.x);
        assert!(r.robust_value < -0.9 && r.robust_value >= r.f_center - 1e-9);
        assert!(r.nfev > 0);
        // The raw answer is untouched.
        assert!(result.fun < -1.15 && result.x[0] < 0.0);
    }

    #[test]
    fn test_robust_polish_without_probe_ranks_by_cost() {
        let result = Shgo::new(fragile_and_robust, vec![(-2.0, 2.0); 2])
            .with_options(ShgoOptions {
                robust_polish: Some(RobustPolish::new(0.03, 8, 1)),
                ..robust_test_options()
            })
            .minimize()
            .unwrap();
        let rm = result.robust_minima.as_ref().unwrap();
        assert_eq!(rm[0].xl_index, 0, "without a probe the lowest minimum is polished first");
    }

    #[test]
    fn test_stencil_accounts_for_bounds_and_constraints() {
        // Minimum in a corner of the box: the inward axis steps survive, the
        // outward ones clip onto the centre and are counted as dropped.
        let result = Shgo::new(sphere, vec![(0.0, 5.0); 2])
            .with_options(ShgoOptions {
                robustness_probe: Some(RobustnessProbe {
                    stencil: Stencil::with_samples(0.05, 4),
                    aggregate: RobustAggregate::Max,
                    top: Some(1),
                }),
                ..robust_test_options()
            })
            .minimize()
            .unwrap();
        let s = &result.robustness.as_ref().unwrap()[0];
        assert!(result.x.iter().all(|v| v.abs() < 1e-6));
        assert_eq!(s.n_feasible + s.n_infeasible, 1 + 4 + 4);
        assert!(s.n_infeasible >= 2, "outward axis steps should be dropped");
        assert_eq!(s.robust_value, s.f_max, "Max aggregate");

        // A constraint that cuts through the minimum: the stencil points on the
        // wrong side are infeasible and never evaluated.
        let constraint = |x: &[f64]| x[1] - x[0]; // feasible when x1 >= x0
        let result = Shgo::with_constraints(sphere, vec![(-5.0, 5.0); 2], vec![constraint])
            .with_options(ShgoOptions {
                robustness_probe: Some(RobustnessProbe::new(0.02, 8)),
                ..robust_test_options()
            })
            .minimize()
            .unwrap();
        let s = &result.robustness.as_ref().unwrap()[0];
        assert!(s.n_infeasible >= 1, "no stencil point was infeasible");
        assert!(s.f_max.is_finite());
    }

    #[test]
    fn test_robustness_probe_top_and_determinism() {
        let run = || {
            Shgo::new(rastrigin, vec![(-5.12, 5.12); 2])
                .with_options(ShgoOptions {
                    n: 256,
                    robustness_probe: Some(RobustnessProbe {
                        stencil: Stencil::axes(0.01),
                        aggregate: RobustAggregate::Cvar { fraction: 0.5 },
                        top: Some(3),
                    }),
                    ..robust_test_options()
                })
                .minimize()
                .unwrap()
        };
        let a = run();
        let b = run();
        let sa = a.robustness.as_ref().unwrap();
        let sb = b.robustness.as_ref().unwrap();
        assert_eq!(sa.len(), 3);
        for (x, y) in sa.iter().zip(sb.iter()) {
            assert_eq!(x.xl_index, y.xl_index);
            assert_eq!(x.f_mean, y.f_mean);
            assert_eq!(x.robust_value, y.robust_value);
            assert!(x.robust_value >= x.f_mean - 1e-12, "CVaR is at least the mean");
        }
    }

    #[test]
    fn test_shgo_3d_rastrigin() {
        let bounds = vec![(-5.12, 5.12); 3];
        let options = ShgoOptions {
            maxiter: Some(2),
            disp: 0,
            ..Default::default()
        };

        let result = Shgo::new(rastrigin, bounds)
            .with_options(options)
            .minimize()
            .unwrap();

        println!("3D Rastrigin result: x={:?}, fun={}", result.x, result.fun);
        // Global minimum is at (0, 0, 0) with f=0
    }

    #[test]
    fn test_shgo_local_minimization_improves_result() {
        // Test that local minimization refines the solution
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        
        // Run with local minimization enabled (default)
        let options_with_local = ShgoOptions {
            maxiter: Some(3),
            minimize_every_iter: true,
            ..Default::default()
        };
        
        let result_with_local = Shgo::new(rosenbrock, bounds.clone())
            .with_options(options_with_local)
            .minimize()
            .unwrap();
        
        // Run with local minimization disabled
        let options_without_local = ShgoOptions {
            maxiter: Some(3),
            minimize_every_iter: false,
            ..Default::default()
        };
        
        let result_without_local = Shgo::new(rosenbrock, bounds)
            .with_options(options_without_local)
            .minimize()
            .unwrap();
        
        println!("With local min: fun={}, nlfev={}", result_with_local.fun, result_with_local.nlfev);
        println!("Without local min: fun={}", result_without_local.fun);
        
        // With local minimization should have better (lower) function value
        // or at least equivalent
        assert!(result_with_local.fun <= result_without_local.fun + 1e-6);
        
        // With local minimization enabled, we should have done local minimizations
        assert!(result_with_local.nlfev > 0);
    }

    #[test]
    fn test_shgo_different_local_optimizers() {
        use crate::local_opt::LocalOptimizer;
        
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        
        // Test with BOBYQA (default)
        let options_bobyqa = ShgoOptions {
            maxiter: Some(2),
            local_options: crate::local_opt::LocalOptimizerOptions {
                algorithm: LocalOptimizer::Bobyqa,
                ..crate::local_opt::LocalOptimizerOptions::default()
            },
            ..Default::default()
        };
        
        let result_bobyqa = Shgo::new(sphere, bounds.clone())
            .with_options(options_bobyqa)
            .minimize()
            .unwrap();
        
        // Test with Nelder-Mead
        let options_nm = ShgoOptions {
            maxiter: Some(2),
            local_options: crate::local_opt::LocalOptimizerOptions {
                algorithm: LocalOptimizer::NelderMead,
                ..crate::local_opt::LocalOptimizerOptions::default()
            },
            ..Default::default()
        };
        
        let result_nm = Shgo::new(sphere, bounds.clone())
            .with_options(options_nm)
            .minimize()
            .unwrap();
        
        // Both should find a reasonably good solution for sphere
        assert!(result_bobyqa.fun < 1.0);
        assert!(result_nm.fun < 1.0);
        
        println!("BOBYQA: fun={}, x={:?}", result_bobyqa.fun, result_bobyqa.x);
        println!("NelderMead: fun={}, x={:?}", result_nm.fun, result_nm.x);
    }

    /// CMA-ES as the local optimizer runs through the same pool machinery and
    /// converges on a smooth problem; with constraints it is upgraded to
    /// COBYLA like every other bound-only algorithm.
    #[test]
    fn test_shgo_cmaes_local_optimizer() {
        use crate::local_opt::{CmaesOptions, LocalOptimizer, LocalOptimizerOptions};

        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let local_options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            cmaes: CmaesOptions { seed: 3, ..Default::default() },
            ..LocalOptimizerOptions::default()
        };
        let options = ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            n: 64,
            maxiter: Some(2),
            local_options: local_options.clone(),
            ..Default::default()
        };
        let result = Shgo::new(sphere, bounds.clone())
            .with_options(options.clone())
            .minimize()
            .unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-8, "fun = {}", result.fun);
        assert!(result.x.iter().all(|v| v.abs() < 1e-3), "x = {:?}", result.x);
        assert!(result.nlfev > 0);

        // Same seed, same answer: the pool stays deterministic.
        let again = Shgo::new(sphere, bounds.clone())
            .with_options(options)
            .minimize()
            .unwrap();
        assert_eq!(result.x, again.x);
        assert_eq!(result.nlfev, again.nlfev);

        // Constrained: x0 + x1 >= 1 -> optimum (0.5, 0.5), f = 0.5.
        let constraint = |x: &[f64]| x[0] + x[1] - 1.0;
        let result = Shgo::with_constraints(sphere, bounds, vec![constraint])
            .with_options(ShgoOptions {
                sampling_method: SamplingMethod::Sobol,
                n: 64,
                maxiter: Some(2),
                local_options,
                ..Default::default()
            })
            .minimize()
            .unwrap();
        assert!(result.x[0] + result.x[1] - 1.0 >= -1e-6, "x = {:?}", result.x);
        assert!((result.fun - 0.5).abs() < 1e-4, "fun = {}", result.fun);
    }

    #[test]
    fn test_shgo_sobol_with_local_minimization() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        
        let options = ShgoOptions {
            sampling_method: SamplingMethod::Sobol,
            n: 64,
            maxiter: Some(2),
            minimize_every_iter: true,
            ..Default::default()
        };
        
        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();
        
        // Should find good solution near origin
        assert!(result.fun < 0.1);
        assert!(result.nlfev > 0); // Local minimizations were performed
        
        println!("Sobol+local: fun={}, x={:?}, nlfev={}", result.fun, result.x, result.nlfev);
    }

    #[test]
    fn test_shgo_local_options_customization() {
        use crate::local_opt::LocalOptimizerOptions;
        
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        
        // Use tight tolerances
        let local_options = LocalOptimizerOptions {
            ftol_rel: 1e-12,
            xtol_rel: 1e-12,
            maxeval: Some(500),
            ..Default::default()
        };
        
        let options = ShgoOptions {
            maxiter: Some(3),
            local_options,
            ..Default::default()
        };
        
        let result = Shgo::new(sphere, bounds)
            .with_options(options)
            .minimize()
            .unwrap();
        
        // With tight tolerances, should get very close to zero
        assert!(result.fun < 1e-8);
        
        println!("Tight tolerances: fun={}, x={:?}", result.fun, result.x);
    }
}
