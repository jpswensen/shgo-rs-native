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
//! use shgo_rs::{Shgo, ShgoOptions, Bounds};
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

/// Type alias for objective function.
pub type ObjectiveFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

/// Type alias for constraint function (g(x) >= 0).
pub type ConstraintFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

/// Type alias for bounds specification.
pub type Bounds = Vec<(f64, f64)>;

/// Sampling method for generating vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMethod {
    /// Simplicial sampling using cyclic product triangulation.
    /// Best for low-dimensional problems (n < 10).
    Simplicial,
    /// Sobol sequence sampling with optional Delaunay triangulation.
    /// Better for high-dimensional problems.
    Sobol,
}

impl Default for SamplingMethod {
    fn default() -> Self {
        SamplingMethod::Simplicial
    }
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

    /// Minimum fraction of feasible vertices required.
    /// Default: 0.5
    pub min_feasible_ratio: f64,

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

    /// Whether to use symmetry exploitation.
    /// Default: true
    pub symmetry: bool,

    /// Verbosity level (0 = silent, 1 = summary, 2 = detailed).
    /// Default: 0
    pub disp: usize,

    /// Number of initial Sobol points to skip.
    /// Default: 0 (include the origin, matching Python's scipy.stats.qmc.Sobol)
    pub sobol_skip: usize,

    /// Infinite bounds replacement value.
    /// Python maps non-finite bounds to ±1e50.
    /// Default: 1e50
    pub infty_constraints: f64,

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
            min_feasible_ratio: 0.5,
            n: 0, // Auto: 2^dim + 1 for simplicial, 128 for sobol
            sampling_method: SamplingMethod::Simplicial,
            minimize_every_iter: true,
            maxiter_local: None,
            symmetry: true,
            disp: 0,
            sobol_skip: 0,
            infty_constraints: 1e50,
            local_options: crate::local_opt::LocalOptimizerOptions {
                algorithm: crate::local_opt::LocalOptimizer::Bobyqa,
                ftol_rel: 1e-12,
                ..crate::local_opt::LocalOptimizerOptions::default()
            },
            workers: None,
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
    /// use shgo_rs::Shgo;
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
    /// use shgo_rs::Shgo;
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

    /// Compute effective iters: if any stopping criterion other than iters
    /// is set, iters becomes None (unlimited). Matches Python behavior.
    fn effective_iters(&self) -> Option<usize> {
        if self.options.maxiter.is_some()
            || self.options.maxfev.is_some()
            || self.options.maxev.is_some()
            || self.options.maxtime.is_some()
            || self.options.f_min.is_some()
        {
            None // Other criteria control termination
        } else {
            self.options.iters
        }
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

        // Reset evaluation counter (but not cancelled flag - allow pre-cancellation)
        self.fev_count.store(0, Ordering::Relaxed);

        // Initialize result
        let mut result = ShgoResult::new(self.dim);

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
        result.nfev = self.fev_count.load(Ordering::Relaxed);
        result.time = start_time.elapsed().as_secs_f64();

        // Sort and deduplicate local minima by function value
        if !result.xl.is_empty() {
            let mut combined: Vec<_> = result.xl.iter().cloned().zip(result.funl.iter().cloned()).collect();
            combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Deduplicate: remove entries where x is within tolerance of an existing entry
            let mut deduped: Vec<(Vec<f64>, f64)> = Vec::new();
            for (x, f) in &combined {
                let is_dup = deduped.iter().any(|(ex, _)| {
                    x.iter().zip(ex.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
                });
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

        // Initial triangulation (with centroid, matching Python)
        complex.triangulate(None, true);
        complex.process_pools();

        let effective_n = self.effective_n();
        let effective_iters = self.effective_iters();
        let mut iteration = 0;

        // Main optimization loop
        loop {
            iteration += 1;
            result.nit = iteration;

            // Refine the complex (with centroids, matching Python)
            if effective_n == 0 || self.options.n == 0 {
                complex.refine_all(true);
            } else {
                complex.refine(Some(effective_n));
            }
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
                let maxiter_local = self.options.maxiter_local.unwrap_or(usize::MAX);

                // Collect candidates: feasible, not already in LMC
                let candidates: Vec<_> = minimizers
                    .iter()
                    .filter(|v| v.feasible() != Some(false))
                    .filter(|v| {
                        !lmap_cache.contains(&Coordinates::new(
                            v.coordinates().as_slice().to_vec(),
                        ))
                    })
                    .take(maxiter_local)
                    .collect();

                if !candidates.is_empty() {
                    // Pre-compute starting points and locally convex bounds
                    // (reads from Complex which is not modified during minimization)
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

                    // Gather successful results (order-independent: results are
                    // deterministic per starting point via lmap_cache dedup)
                    for local_min in local_results.into_iter().flatten() {
                        if local_min.success {
                            result.nlfev += local_min.nfev;
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
            if self.check_stopping_criteria(iteration, effective_iters, start_time.elapsed(), result)? {
                break;
            }
        }

        // Add function evaluations from local minimizations
        result.nfev += lmap_cache.total_fev();

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

            // Create vertices and store in input order for Delaunay index mapping
            let vertices_in_order: Vec<std::sync::Arc<crate::Vertex>> = points
                .iter()
                .map(|p| cache.get_or_create(p.clone()))
                .collect();

            // ---- Build Delaunay connectivity (vf_to_vv) ----
            // Need at least dim+2 non-degenerate points for triangulation
            if points.len() >= self.dim + 2 {
                if self.dim == 1 {
                    // 1D: sort points and connect consecutive pairs
                    // (Delaunay in 1D is just sorted adjacency)
                    let mut sorted_indices: Vec<usize> = (0..points.len()).collect();
                    sorted_indices.sort_by(|&a, &b| {
                        points[a][0]
                            .partial_cmp(&points[b][0])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for w in sorted_indices.windows(2) {
                        crate::Vertex::connect_bidirectional(
                            &vertices_in_order[w[0]],
                            &vertices_in_order[w[1]],
                        );
                    }
                } else {
                    // Multi-dimensional: Delaunay triangulation via QHull
                    // (same backend as Python's scipy.spatial.Delaunay)
                    let qh = qhull::Qh::new_delaunay(
                        points.iter().map(|p| p.iter().cloned()),
                    )
                    .map_err(|e| {
                        ShgoError::MeshGeneration(format!(
                            "Delaunay triangulation failed: {}",
                            e
                        ))
                    })?;

                    // Convert vertex-face mesh to vertex-vertex connectivity
                    // Matches Python's vf_to_vv exactly:
                    //   for s in simplices:
                    //       edges = itertools.combinations(s, self.dim)
                    //       for e in edges: connect(e[0], e[1])
                    // For dim=2 this is equivalent to all-pairs.
                    // For dim>2 it connects only the first two vertices of each
                    // (dim)-combination face, producing fewer edges than all-pairs.
                    let dim = self.dim;
                    for simplex in qh.simplices().filter(|f| !f.upper_delaunay()) {
                        if let Some(verts) = simplex.vertices() {
                            let simplex_indices: Vec<usize> = verts
                                .iter()
                                .filter_map(|v| v.index(&qh))
                                .collect();

                            // combinations(simplex_indices, dim) → connect e[0]-e[1]
                            // This matches Python's itertools.combinations(s, self.dim)
                            if dim >= 2 && simplex_indices.len() >= dim {
                                // Generate combinations of `dim` elements
                                let mut combo = vec![0usize; dim];
                                // Initialize combo to [0, 1, ..., dim-1]
                                for k in 0..dim {
                                    combo[k] = k;
                                }
                                let n = simplex_indices.len();
                                loop {
                                    // Connect the first two elements of this combination
                                    let pi = simplex_indices[combo[0]];
                                    let pj = simplex_indices[combo[1]];
                                    if pi < vertices_in_order.len()
                                        && pj < vertices_in_order.len()
                                    {
                                        crate::Vertex::connect_bidirectional(
                                            &vertices_in_order[pi],
                                            &vertices_in_order[pj],
                                        );
                                    }

                                    // Advance to next combination
                                    let mut i = dim - 1;
                                    loop {
                                        combo[i] += 1;
                                        if combo[i] <= n - dim + i {
                                            break;
                                        }
                                        if i == 0 {
                                            // All combinations exhausted
                                            combo[0] = n; // sentinel
                                            break;
                                        }
                                        i -= 1;
                                    }
                                    if combo[0] >= n {
                                        break;
                                    }
                                    // Reset trailing elements
                                    for j in (i + 1)..dim {
                                        combo[j] = combo[j - 1] + 1;
                                    }
                                }
                            } else if simplex_indices.len() >= 2 {
                                // Fallback for dim=1 or degenerate: pairs
                                let pi = simplex_indices[0];
                                let pj = simplex_indices[1];
                                if pi < vertices_in_order.len()
                                    && pj < vertices_in_order.len()
                                {
                                    crate::Vertex::connect_bidirectional(
                                        &vertices_in_order[pi],
                                        &vertices_in_order[pj],
                                    );
                                }
                            }
                        }
                    }

                    if self.options.disp > 1 {
                        println!(
                            "  Delaunay triangulation: {} simplices, {} vertices",
                            qh.simplices()
                                .filter(|f| !f.upper_delaunay())
                                .count(),
                            qh.num_vertices()
                        );
                    }
                }
            }

            // Process pending evaluations (function values + constraints)
            cache.process_pools();

            // Find minimizers using topological analysis
            // (now with Delaunay connectivity, f(v) < f(all neighbors) works correctly)
            let minimizers = cache.find_all_minimizers();
            
            // Process minimizers with local optimization
            if self.options.minimize_every_iter {
                // Filter to feasible vertices, not in LMC, limited count
                let maxiter_local = self.options.maxiter_local.unwrap_or(usize::MAX);
                let candidates: Vec<_> = minimizers
                    .iter()
                    .filter(|v| v.feasible() != Some(false))
                    .filter(|v| !lmap_cache.contains(&Coordinates::new(v.coordinates().as_slice().to_vec())))
                    .take(maxiter_local)
                    .collect();

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

                for local_min in local_results.into_iter().flatten() {
                    if local_min.success {
                        result.xl.push(local_min.x);
                        result.funl.push(local_min.fun);
                        result.nlfev += local_min.nfev;
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
            if self.check_stopping_criteria(iteration, effective_iters, start_time.elapsed(), result)? {
                break;
            }
        }

        // Add function evaluations from local minimizations
        result.nfev += lmap_cache.total_fev();

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

        // Check sampling evaluation limit (maxev)
        if let Some(maxev) = self.options.maxev {
            if self.fev_count() >= maxev {
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
            let constraint_fns: Vec<Box<dyn Fn(&[f64]) -> f64>> = self
                .constraints
                .iter()
                .map(|c| {
                    let c = Arc::clone(c);
                    Box::new(move |x: &[f64]| c(x)) as Box<dyn Fn(&[f64]) -> f64>
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

        let local_min = LocalMinimum {
            x: result.x,
            fun: result.fun,
            success: result.success,
            nfev: result.nfev,
            nit: result.nit,
        };

        // Cache the result
        cache.insert(coords, local_min.clone());

        Some(local_min)
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
        assert_eq!(result.nfev, shgo.fev_count());
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
