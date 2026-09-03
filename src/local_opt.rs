//! Local optimization interface using NLOPT.
//!
//! This module provides a wrapper around NLOPT algorithms for performing local
//! minimization within the SHGO framework. Since NLOPT's `Nlopt` struct is not
//! `Send` or `Sync`, we create fresh optimizer instances for each local minimization.
//!
//! # Supported Algorithms
//!
//! - **BOBYQA** (default): Bound Optimization BY Quadratic Approximation.
//!   Derivative-free, works well for smooth functions with bound constraints.
//!
//! - **COBYLA**: Constrained Optimization BY Linear Approximation.
//!   Derivative-free, supports nonlinear inequality constraints.
//!
//! - **SLSQP**: Sequential Least Squares Programming.
//!   Gradient-based, supports both equality and inequality constraints.
//!   Gradients are supplied by forward finite differences (SciPy's default
//!   `'2-point'` scheme), evaluated in parallel across coordinates with rayon.
//!
//! - **LBFGS**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno.
//!   Gradient-based (finite-difference gradients as above), good for smooth
//!   unconstrained problems.
//!
//! - **NelderMead**: Nelder-Mead simplex method.
//!   Derivative-free, robust for noisy functions.
//!
//! - **PRAXIS**: Principal Axis method.
//!   Derivative-free, good for smooth functions.
//!
//! - **Sbplx**: Rowan's subplex method (NLopt `LN_SBPLX`).
//!   Nelder-Mead restarted on low-dimensional subspaces; tolerates noise and
//!   non-smoothness and scales to higher dimensions better than Nelder-Mead.
//!
//! - **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy, provided by
//!   the [`cmaes`](https://docs.rs/cmaes) crate rather than NLopt. Population
//!   based and rank based, so it is robust to noise, plateaus, and small-scale
//!   ruggedness that trap model-based methods, at the cost of more evaluations
//!   on smooth problems. The crate has no bound support, so the search space
//!   is mapped onto the box by normalising each coordinate by its bound width
//!   and reflecting out-of-box points back inside (see [`CmaesOptions`]).
//!   Every run is seeded from the starting point, so results are deterministic.
//!
//! # Constraints
//!
//! Inequality constraints come in two forms: closures in SHGO's `g(x) >= 0`
//! convention and explicit [`LinearConstraint`]s (`a · x >= b`). COBYLA and
//! SLSQP take both natively. For every other algorithm
//! [`LocalOptimizerOptions::constraint_handling`] decides what happens:
//! [`ConstraintHandling::UpgradeToCobyla`] (default, SciPy-like) switches the
//! run to COBYLA, [`ConstraintHandling::KeepAlgorithm`] wraps NLopt methods in
//! NLopt's augmented Lagrangian (AUGLAG) so BOBYQA or Subplex keep their
//! identity. CMA-ES handles linear constraints itself by mirroring every
//! sample across a violated constraint, as it does at the bounds (exact, so
//! it is used under both settings), and, under `KeepAlgorithm`, nonlinear closures by a
//! feasibility-first penalty that never evaluates the objective at an
//! infeasible point.
//!
//! # Example
//!
//! ```
//! use shgo::local_opt::{LocalOptimizer, LocalOptimizerOptions, minimize_local};
//!
//! let sphere = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//! let x0 = vec![1.0, 1.0];
//!
//! let options = LocalOptimizerOptions {
//!     algorithm: LocalOptimizer::Bobyqa,
//!     ftol_rel: 1e-8,
//!     xtol_rel: 1e-8,
//!     maxeval: Some(1000),
//!     ..Default::default()
//! };
//!
//! let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
//! println!("Minimum at {:?} with value {}", result.x, result.fun);
//! ```

use cmaes::{CMAESOptions, DVector, TerminationReason};
use nlopt::{Algorithm, Nlopt, Target};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Local optimization algorithm selection.
///
/// These algorithms are available from the NLOPT library. The choice of algorithm
/// depends on the problem characteristics:
///
/// - Use `Bobyqa` (default) for smooth functions with bound constraints
/// - Use `Cobyla` when you have nonlinear inequality constraints
/// - Use `Slsqp` when you need gradients and have constraints
/// - Use `Lbfgs` for smooth unconstrained problems with gradients
/// - Use `NelderMead` or `Sbplx` for noisy or non-smooth functions
/// - Use `Cmaes` for rugged, noisy, or plateau-ridden basins where the
///   quadratic-model methods stall (see [`CmaesOptions`])
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalOptimizer {
    /// Bound Optimization BY Quadratic Approximation.
    /// Derivative-free, bounds only. Default choice.
    #[default]
    Bobyqa,

    /// Constrained Optimization BY Linear Approximation.
    /// Derivative-free, supports nonlinear inequality constraints.
    Cobyla,

    /// Sequential Least Squares Programming.
    /// Gradient-based (finite-difference gradients), supports inequality
    /// constraints.
    Slsqp,

    /// Limited-memory BFGS.
    /// Gradient-based (finite-difference gradients), bounds only.
    Lbfgs,

    /// Nelder-Mead simplex method.
    /// Derivative-free, robust for noisy functions.
    NelderMead,

    /// Principal Axis method.
    /// Derivative-free, good for smooth functions.
    Praxis,

    /// NEWUOA with bounds.
    /// Derivative-free, similar to BOBYQA but may work better for some problems.
    NewuoaBound,

    /// Sbplx (subplex) method.
    /// Derivative-free variant of Nelder-Mead for higher dimensions.
    Sbplx,

    /// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) from the
    /// `cmaes` crate. Derivative-free, population based; bounds are enforced
    /// by reflection, constraints are not supported (SHGO upgrades to COBYLA).
    /// Tuned via [`LocalOptimizerOptions::cmaes`].
    Cmaes,
}

impl LocalOptimizer {
    /// Convert to the NLOPT Algorithm enum. Returns `None` for algorithms
    /// that are not provided by NLopt (`Cmaes`).
    pub fn to_nlopt_algorithm(self) -> Option<Algorithm> {
        Some(match self {
            LocalOptimizer::Bobyqa => Algorithm::Bobyqa,
            LocalOptimizer::Cobyla => Algorithm::Cobyla,
            LocalOptimizer::Slsqp => Algorithm::Slsqp,
            LocalOptimizer::Lbfgs => Algorithm::Lbfgs,
            LocalOptimizer::NelderMead => Algorithm::Neldermead,
            LocalOptimizer::Praxis => Algorithm::Praxis,
            LocalOptimizer::NewuoaBound => Algorithm::NewuoaBound,
            LocalOptimizer::Sbplx => Algorithm::Sbplx,
            LocalOptimizer::Cmaes => return None,
        })
    }

    /// Whether the algorithm is implemented by NLopt (everything except
    /// `Cmaes`).
    pub fn is_nlopt(self) -> bool {
        self.to_nlopt_algorithm().is_some()
    }

    /// Check if the algorithm supports nonlinear constraints.
    pub fn supports_constraints(self) -> bool {
        matches!(self, LocalOptimizer::Cobyla | LocalOptimizer::Slsqp)
    }

    /// Check if the algorithm supports [`LinearConstraint`]s natively
    /// (COBYLA and SLSQP as generic constraints, CMA-ES by mirroring).
    pub fn supports_linear_constraints(self) -> bool {
        matches!(
            self,
            LocalOptimizer::Cobyla | LocalOptimizer::Slsqp | LocalOptimizer::Cmaes
        )
    }

    /// Whether the algorithm can run a problem with these constraint kinds
    /// without upgrading or wrapping.
    pub fn handles_natively(self, has_nonlinear: bool, has_linear: bool) -> bool {
        (!has_nonlinear || self.supports_constraints())
            && (!has_linear || self.supports_linear_constraints())
    }

    /// Check if the algorithm requires gradients (supplied by finite differences).
    pub fn requires_gradient(self) -> bool {
        matches!(self, LocalOptimizer::Slsqp | LocalOptimizer::Lbfgs)
    }
}

/// What to do with constraints when the chosen algorithm has no native
/// support for them (everything except COBYLA and SLSQP; CMA-ES supports
/// linear constraints natively by projection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstraintHandling {
    /// Switch the run to COBYLA, as SciPy's SHGO effectively does. Default.
    #[default]
    UpgradeToCobyla,

    /// Keep the chosen algorithm. NLopt methods are wrapped in NLopt's
    /// augmented Lagrangian (`AUGLAG`), which folds the constraints into an
    /// adaptive penalty and re-runs the inner method until the multipliers
    /// converge; the result satisfies the constraints to `constraint_tol`.
    /// CMA-ES mirrors samples across linear constraints and treats nonlinear
    /// closures with a feasibility-first penalty: an infeasible sample is not
    /// evaluated and ranks below the start point by its total violation.
    KeepAlgorithm,
}

/// Linear inequality constraint `a · x >= b` (SHGO's `g(x) >= 0` convention
/// with `g(x) = a · x - b`).
///
/// Declaring a constraint as linear rather than as a closure lets CMA-ES
/// keep every sample feasible by mirroring it across the constraint (the
/// same fold it applies at the bounds) instead of falling back to COBYLA,
/// and lets SHGO's sampling filter it without calling user code.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearConstraint {
    /// Coefficient vector `a` (one entry per dimension).
    pub a: Vec<f64>,
    /// Right-hand side `b`.
    pub b: f64,
}

impl LinearConstraint {
    /// `a · x >= b`.
    pub fn ge(a: Vec<f64>, b: f64) -> Self {
        Self { a, b }
    }

    /// `a · x <= b`, stored as `-a · x >= -b`.
    pub fn le(a: Vec<f64>, b: f64) -> Self {
        Self {
            a: a.into_iter().map(|v| -v).collect(),
            b: -b,
        }
    }

    /// `g(x) = a · x - b`; feasible when `>= 0`.
    pub fn value(&self, x: &[f64]) -> f64 {
        self.a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() - self.b
    }

    /// Whether `x` satisfies the constraint to within `tol`.
    pub fn is_satisfied(&self, x: &[f64], tol: f64) -> bool {
        self.value(x) >= -tol
    }
}

/// Which algorithm actually runs for a problem with the given kinds of
/// constraints, after applying `options.constraint_handling`. Exposed so
/// callers (SHGO) can warn when the choice differs from `options.algorithm`.
pub fn effective_algorithm(
    options: &LocalOptimizerOptions,
    has_nonlinear: bool,
    has_linear: bool,
) -> LocalOptimizer {
    let algo = options.algorithm;
    if algo.handles_natively(has_nonlinear, has_linear) {
        return algo;
    }
    match options.constraint_handling {
        ConstraintHandling::UpgradeToCobyla => LocalOptimizer::Cobyla,
        ConstraintHandling::KeepAlgorithm => algo,
    }
}

/// Options specific to the CMA-ES backend (`LocalOptimizer::Cmaes`).
///
/// CMA-ES searches an unbounded space; this wrapper normalises every
/// coordinate by its bound width (so one step size serves all coordinates)
/// and reflects sampled points back into the box (a triangle-wave fold), so
/// every objective evaluation is in bounds and a minimum on the boundary is
/// reachable without the flat plateaus that clipping would create. The
/// shared tolerances of [`LocalOptimizerOptions`] map as follows: `ftol_abs`
/// bounds the function-value spread (`tol_fun`/`tol_fun_hist`), `xtol_rel`
/// and `xtol_abs` bound the distribution's standard deviation in normalised
/// units (`tol_x`), `maxeval` and `maxtime` cap the run. `initial_step` is an
/// NLopt setting and is ignored here; use `sigma0` instead.
#[derive(Debug, Clone)]
pub struct CmaesOptions {
    /// Initial step size (`sigma0`) as a fraction of each coordinate's bound
    /// width (raw units for a coordinate without finite bounds). Larger values
    /// step over more small-scale structure; smaller values converge faster on
    /// smooth basins. Default: 0.25.
    pub sigma0: f64,

    /// Population size per generation (`lambda`). `None` uses the CMA-ES
    /// default `4 + floor(3 ln(dim))`. Default: `None`.
    pub population_size: Option<usize>,

    /// Base RNG seed. The seed of each run is derived from this value and the
    /// starting point, so a given start always reproduces the same result and
    /// the parallel minimizer pool stays deterministic. Default: 0.
    pub seed: u64,

    /// Evaluate each generation's population in parallel on the rayon pool.
    /// Useful when the objective is expensive and the candidate pool is
    /// narrower than the core count; otherwise it only adds scheduling on top
    /// of SHGO's per-candidate parallelism. Default: `false`.
    pub parallel_eval: bool,

    /// After termination also evaluate the final distribution mean (one extra
    /// evaluation) and return it if it beats the best sampled point.
    /// Default: `true`.
    pub eval_final_mean: bool,
}

impl Default for CmaesOptions {
    fn default() -> Self {
        Self {
            sigma0: 0.25,
            population_size: None,
            seed: 0,
            parallel_eval: false,
            eval_final_mean: true,
        }
    }
}

/// Options for local optimization.
#[derive(Debug, Clone)]
pub struct LocalOptimizerOptions {
    /// Algorithm to use.
    pub algorithm: LocalOptimizer,

    /// Relative tolerance on function value.
    /// Stop when |f_new - f_old| < ftol_rel * |f_old|.
    pub ftol_rel: f64,

    /// Absolute tolerance on function value.
    /// Stop when |f_new - f_old| < ftol_abs.
    pub ftol_abs: f64,

    /// Relative tolerance on optimization parameters.
    /// Stop when all |x_new - x_old| < xtol_rel * |x_old|.
    pub xtol_rel: f64,

    /// Absolute tolerance on optimization parameters.
    /// Stop when all |x_new - x_old| < xtol_abs.
    pub xtol_abs: f64,

    /// Maximum number of function evaluations (finite-difference gradient
    /// evaluations count towards this budget, as in SciPy).
    pub maxeval: Option<u32>,

    /// Maximum time in seconds.
    pub maxtime: Option<f64>,

    /// Initial step size for derivative-free methods.
    /// If None, NLOPT chooses heuristically.
    pub initial_step: Option<f64>,

    /// Constraint tolerance (for algorithms that support constraints).
    pub constraint_tol: f64,

    /// Verbosity: if true, print cost at each function evaluation during local optimization.
    pub disp: bool,

    /// Settings used only when `algorithm == LocalOptimizer::Cmaes`.
    pub cmaes: CmaesOptions,

    /// What to do with constraints the chosen algorithm cannot take natively.
    /// Default: `UpgradeToCobyla`.
    pub constraint_handling: ConstraintHandling,
}

impl Default for LocalOptimizerOptions {
    fn default() -> Self {
        Self {
            algorithm: LocalOptimizer::Bobyqa,
            ftol_rel: 1e-8,
            ftol_abs: 1e-14,
            xtol_rel: 1e-8,
            xtol_abs: 1e-14,
            maxeval: Some(1000),
            maxtime: None,
            initial_step: None,
            constraint_tol: 1e-8,
            disp: false,
            cmaes: CmaesOptions::default(),
            constraint_handling: ConstraintHandling::UpgradeToCobyla,
        }
    }
}

/// Boxed inequality-constraint function (`g(x) >= 0` means feasible).
pub type BoxedConstraint = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// A scalar function reference usable from the NLOPT callbacks (and from the
/// rayon workers that evaluate finite-difference gradients).
type ScalarFn<'a> = &'a (dyn Fn(&[f64]) -> f64 + Sync);

/// An owned scalar function (linear constraints turned into closures).
type BoxedScalarFn = Box<dyn Fn(&[f64]) -> f64 + Sync>;

/// Result of a local minimization.
#[derive(Debug, Clone)]
pub struct LocalOptResult {
    /// Location of the local minimum.
    pub x: Vec<f64>,
    /// Function value at the minimum.
    pub fun: f64,
    /// Whether the local minimization succeeded.
    pub success: bool,
    /// Status message.
    pub message: String,
    /// Number of function evaluations used (including finite-difference
    /// gradient evaluations).
    pub nfev: usize,
    /// Number of iterations (not always available).
    pub nit: usize,
}

impl LocalOptResult {
    /// Create a failed result.
    fn failure(x0: &[f64], message: String) -> Self {
        Self {
            x: x0.to_vec(),
            fun: f64::INFINITY,
            success: false,
            message,
            nfev: 0,
            nit: 0,
        }
    }
}

/// Forward finite-difference gradient of `f` at `x` (SciPy's `'2-point'`
/// scheme: relative step `sqrt(eps) * max(1, |x_i|)`, stepping backwards when a
/// forward step would leave the bounds). The `dim` perturbed evaluations run in
/// parallel on the rayon pool. Returns the number of function evaluations made.
fn fd_gradient(
    f: ScalarFn<'_>,
    x: &[f64],
    fx: f64,
    bounds: &[(f64, f64)],
    grad: &mut [f64],
) -> usize {
    let rel_step = f64::EPSILON.sqrt();
    grad.par_iter_mut().enumerate().for_each(|(i, g)| {
        let mut h = rel_step * x[i].abs().max(1.0);
        let (lb, ub) = bounds.get(i).copied().unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        // Prefer a forward step; fall back to a backward step at the upper bound.
        if x[i] + h > ub && x[i] - h >= lb {
            h = -h;
        }
        let mut xp = x.to_vec();
        xp[i] += h;
        // Recompute the actual step (guards against rounding for tiny |h|).
        let actual = xp[i] - x[i];
        let fp = f(&xp);
        *g = if actual != 0.0 { (fp - fx) / actual } else { 0.0 };
    });
    x.len()
}

/// Run one local minimization. Shared by the public entry points.
///
/// `constraints` are inequality constraints in SHGO's `g(x) >= 0` convention
/// and `linear` are explicit linear ones. If any are present and
/// `options.algorithm` cannot take them natively, `options.constraint_handling`
/// decides between upgrading to COBYLA and keeping the algorithm (AUGLAG for
/// NLopt methods, projection/penalty for CMA-ES). Gradient-based algorithms
/// receive forward-difference gradients of the objective and of every
/// constraint. `Cmaes` is dispatched to the `cmaes` crate; everything else
/// runs through NLopt.
fn run_local(
    func: ScalarFn<'_>,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: &[ScalarFn<'_>],
    linear: &[LinearConstraint],
    options: &LocalOptimizerOptions,
) -> LocalOptResult {
    let dim = x0.len();

    // Validate dimensions
    if bounds.len() != dim {
        return LocalOptResult::failure(
            x0,
            format!(
                "Dimension mismatch: x0 has {} elements but bounds has {}",
                dim,
                bounds.len()
            ),
        );
    }
    if let Some(bad) = linear.iter().find(|c| c.a.len() != dim) {
        return LocalOptResult::failure(
            x0,
            format!(
                "Dimension mismatch: linear constraint has {} coefficients but x0 has {}",
                bad.a.len(),
                dim
            ),
        );
    }

    let has_nonlinear = !constraints.is_empty();
    let has_linear = !linear.is_empty();
    let algo = effective_algorithm(options, has_nonlinear, has_linear);

    match algo.to_nlopt_algorithm() {
        Some(nlopt_algo) => {
            // NLopt sees linear constraints as ordinary inequality closures.
            let linear_fns: Vec<BoxedScalarFn> = linear
                .iter()
                .map(|c| {
                    let c = c.clone();
                    Box::new(move |x: &[f64]| c.value(x)) as BoxedScalarFn
                })
                .collect();
            let all: Vec<ScalarFn<'_>> = constraints
                .iter()
                .copied()
                .chain(linear_fns.iter().map(|b| &**b as ScalarFn<'_>))
                .collect();
            let use_auglag = !algo.handles_natively(has_nonlinear, has_linear);
            run_nlopt(func, x0, bounds, &all, options, algo, nlopt_algo, use_auglag)
        }
        None => run_cmaes(func, x0, bounds, constraints, linear, options),
    }
}

/// Run one NLOPT local minimization with the already-resolved algorithm.
/// With `use_auglag` the algorithm runs as the subsidiary optimizer of
/// NLopt's augmented Lagrangian, which is how bound-only methods take the
/// constraints.
#[allow(clippy::too_many_arguments)]
fn run_nlopt(
    func: ScalarFn<'_>,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: &[ScalarFn<'_>],
    options: &LocalOptimizerOptions,
    algo: LocalOptimizer,
    nlopt_algo: Algorithm,
    use_auglag: bool,
) -> LocalOptResult {
    let dim = x0.len();
    let needs_grad = algo.requires_gradient();
    let label = if use_auglag {
        format!("AUGLAG({:?})", algo)
    } else {
        format!("{:?}", algo)
    };

    // Track function evaluations (finite-difference evaluations included).
    let fev_count = AtomicUsize::new(0);
    let disp = options.disp;

    // Objective wrapper for NLOPT.
    // Signature: (&[f64], Option<&mut [f64]>, &mut UserData) -> f64
    let objective = |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let n = fev_count.fetch_add(1, Ordering::Relaxed) + 1;
        let val = func(x);
        if disp {
            println!("  [{} eval #{:>3}] f = {:+.6e}", label, n, val);
        }
        if let Some(g) = grad {
            if needs_grad {
                let used = fd_gradient(func, x, val, bounds, g);
                fev_count.fetch_add(used, Ordering::Relaxed);
            }
        }
        val
    };

    // Create the NLOPT optimizer: either the algorithm itself, or AUGLAG with
    // the algorithm as its subsidiary (NLopt copies the subsidiary's
    // algorithm and stopping criteria; its bounds/objective are ignored).
    let mut opt = if use_auglag {
        let mut outer = Nlopt::new(Algorithm::Auglag, dim, objective, Target::Minimize, ());
        let mut inner = outer.get_local_optimizer(nlopt_algo);
        let _ = inner.set_ftol_rel(options.ftol_rel);
        let _ = inner.set_ftol_abs(options.ftol_abs);
        let _ = inner.set_xtol_rel(options.xtol_rel);
        let _ = inner.set_xtol_abs1(options.xtol_abs);
        if let Some(maxeval) = options.maxeval {
            let _ = inner.set_maxeval(maxeval);
        }
        if let Some(step) = options.initial_step {
            let _ = inner.set_initial_step1(step);
        }
        if outer.set_local_optimizer(inner).is_err() {
            return LocalOptResult::failure(
                x0,
                format!("Failed to set {:?} as the AUGLAG subsidiary optimizer", algo),
            );
        }
        outer
    } else {
        Nlopt::new(nlopt_algo, dim, objective, Target::Minimize, ())
    };

    // Set bounds
    let lower_bounds: Vec<f64> = bounds.iter().map(|(l, _)| *l).collect();
    let upper_bounds: Vec<f64> = bounds.iter().map(|(_, u)| *u).collect();

    if opt.set_lower_bounds(&lower_bounds).is_err() {
        return LocalOptResult::failure(x0, "Failed to set lower bounds".to_string());
    }
    if opt.set_upper_bounds(&upper_bounds).is_err() {
        return LocalOptResult::failure(x0, "Failed to set upper bounds".to_string());
    }

    // Add constraints.
    // SHGO uses g(x) >= 0 (feasible), NLOPT uses fc(x) <= 0 (feasible), so
    // register -g(x) <= 0. Gradient-based algorithms also get d(-g)/dx by
    // finite differences (constraint evaluations are not counted in nfev,
    // matching SciPy which reports objective evaluations only).
    for &g_fn in constraints {
        let negated = move |x: &[f64]| -g_fn(x);
        let constraint_wrapper = move |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
            let v = negated(x);
            if let Some(gr) = grad {
                if needs_grad {
                    fd_gradient(&negated, x, v, bounds, gr);
                }
            }
            v
        };
        if opt
            .add_inequality_constraint(constraint_wrapper, (), options.constraint_tol)
            .is_err()
        {
            return LocalOptResult::failure(x0, "Failed to add constraint".to_string());
        }
    }

    // Set tolerances
    let _ = opt.set_ftol_rel(options.ftol_rel);
    let _ = opt.set_ftol_abs(options.ftol_abs);
    let _ = opt.set_xtol_rel(options.xtol_rel);
    let _ = opt.set_xtol_abs1(options.xtol_abs);

    // Set evaluation limits
    if let Some(maxeval) = options.maxeval {
        let _ = opt.set_maxeval(maxeval);
    }
    if let Some(maxtime) = options.maxtime {
        let _ = opt.set_maxtime(maxtime);
    }

    // Set initial step if specified
    if let Some(step) = options.initial_step {
        let _ = opt.set_initial_step1(step);
    }

    // Run optimization
    let mut x = x0.to_vec();
    let result = opt.optimize(&mut x);

    let final_fev = fev_count.load(Ordering::Relaxed);

    if disp {
        match &result {
            Ok((state, fval)) => println!(
                "  [{}] {:?}: f_best = {:+.6e} ({} evals)",
                label, state, fval, final_fev
            ),
            Err((state, fval)) => println!(
                "  [{}] FAILED {:?}: f_best = {:+.6e} ({} evals)",
                label, state, fval, final_fev
            ),
        }
    }

    match result {
        // NLOPT reports a success state even when the objective is non-finite
        // everywhere it looked (it then just stops on xtol); do not call that
        // a success. `fun` is returned as-is so callers can see what happened.
        Ok((success_state, fval)) if !fval.is_finite() => LocalOptResult {
            x,
            fun: fval,
            success: false,
            message: format!(
                "Optimization failed: objective is non-finite at the returned point ({:?})",
                success_state
            ),
            nfev: final_fev,
            nit: 0,
        },
        Ok((success_state, fval)) => LocalOptResult {
            x,
            fun: fval,
            success: true,
            message: format!("Optimization succeeded: {:?} ({})", success_state, label),
            nfev: final_fev,
            nit: 0, // NLOPT doesn't track iterations for all algorithms
        },
        Err((fail_state, fval)) => {
            // With finite-difference gradients, NLOPT's line search stops
            // being able to make progress once the objective change per step
            // drops below the gradient noise — which happens precisely at
            // convergence (e.g. L-BFGS on a quadratic reaches f ~ 1e-20 and
            // then reports a generic `Failure`). `x`/`fval` hold the best
            // point found, so report that as converged rather than as an
            // error. Genuine problems (invalid arguments, bad bounds) still
            // fail before any evaluation and are reported as failures.
            let stalled_at_precision = needs_grad
                && matches!(
                    fail_state,
                    nlopt::FailState::Failure | nlopt::FailState::RoundoffLimited
                )
                && fval.is_finite()
                && final_fev > 1;
            if stalled_at_precision {
                LocalOptResult {
                    x,
                    fun: fval,
                    success: true,
                    message: format!(
                        "Optimization succeeded: line search stalled at finite-difference precision ({:?})",
                        fail_state
                    ),
                    nfev: final_fev,
                    nit: 0,
                }
            } else {
                LocalOptResult {
                    x,
                    fun: fval,
                    success: false,
                    message: format!("Optimization failed: {:?}", fail_state),
                    nfev: final_fev,
                    nit: 0,
                }
            }
        }
    }
}

/// Fold a value into `[lo, hi]` by reflection (a triangle wave with period
/// `2 (hi - lo)`), so the map is continuous, distance preserving inside the
/// box, and reaches the bounds exactly. One-sided bounds reflect about the
/// finite side; unbounded coordinates pass through.
fn reflect_into(v: f64, lo: f64, hi: f64) -> f64 {
    match (lo.is_finite(), hi.is_finite()) {
        (true, true) if hi > lo => {
            let w = hi - lo;
            let t = (v - lo).rem_euclid(2.0 * w);
            lo + if t <= w { t } else { 2.0 * w - t }
        }
        (true, true) => lo,
        (true, false) => lo + (v - lo).abs(),
        (false, true) => hi - (hi - v).abs(),
        (false, false) => v,
    }
}

/// Maps CMA-ES's unbounded, normalised search space onto the bound box:
/// `x_i = reflect(lo_i + z_i * width_i)`, with `width_i = 1` for coordinates
/// without two finite bounds.
struct BoxMap {
    lo: Vec<f64>,
    hi: Vec<f64>,
    scale: Vec<f64>,
}

impl BoxMap {
    fn new(bounds: &[(f64, f64)]) -> Self {
        let lo: Vec<f64> = bounds.iter().map(|b| b.0).collect();
        let hi: Vec<f64> = bounds.iter().map(|b| b.1).collect();
        let scale = bounds
            .iter()
            .map(|&(l, u)| {
                if l.is_finite() && u.is_finite() && u > l {
                    u - l
                } else {
                    1.0
                }
            })
            .collect();
        Self { lo, hi, scale }
    }

    fn origin(&self, i: usize) -> f64 {
        if self.lo[i].is_finite() {
            self.lo[i]
        } else {
            0.0
        }
    }

    /// Box coordinates -> normalised search coordinates (identity up to
    /// scaling; `x` is assumed to lie inside the box).
    fn to_internal(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| (xi - self.origin(i)) / self.scale[i])
            .collect()
    }

    /// The box in normalised units (`±inf` where a bound is missing).
    fn u_bounds(&self) -> Vec<(f64, f64)> {
        (0..self.lo.len())
            .map(|i| {
                let o = self.origin(i);
                let lo = if self.lo[i].is_finite() { (self.lo[i] - o) / self.scale[i] } else { f64::NEG_INFINITY };
                let hi = if self.hi[i].is_finite() { (self.hi[i] - o) / self.scale[i] } else { f64::INFINITY };
                (lo, hi)
            })
            .collect()
    }

    /// Fold unbounded search coordinates into the box (reflection), in
    /// normalised units.
    fn fold(&self, z: &[f64]) -> Vec<f64> {
        let ub = self.u_bounds();
        z.iter()
            .zip(&ub)
            .map(|(&zi, &(lo, hi))| reflect_into(zi, lo, hi))
            .collect()
    }

    /// Normalised (in-box) coordinates -> box coordinates.
    fn unscale(&self, u: &[f64]) -> Vec<f64> {
        u.iter()
            .enumerate()
            .map(|(i, &ui)| self.origin(i) + ui * self.scale[i])
            .collect()
    }

    /// Linear constraints `a · x >= b` rewritten in normalised units:
    /// `(a ∘ scale) · u >= b - a · origin`.
    fn half_spaces(&self, linear: &[LinearConstraint]) -> Vec<HalfSpace> {
        linear
            .iter()
            .map(|c| {
                let a: Vec<f64> = c.a.iter().zip(&self.scale).map(|(ai, si)| ai * si).collect();
                let shift: f64 = c.a.iter().enumerate().map(|(i, ai)| ai * self.origin(i)).sum();
                let aa = a.iter().map(|v| v * v).sum();
                HalfSpace { a, b: c.b - shift, aa }
            })
            .collect()
    }

    /// Smallest normalisation factor (for converting absolute x tolerances).
    fn min_scale(&self) -> f64 {
        self.scale.iter().cloned().fold(f64::INFINITY, f64::min).max(f64::MIN_POSITIVE)
    }
}

/// Half-space `a · u >= b` in normalised coordinates (`aa = a · a`).
struct HalfSpace {
    a: Vec<f64>,
    b: f64,
    aa: f64,
}

/// Euclidean projection of `u` onto `box ∩ half-spaces` by Dykstra's
/// alternating projections (exact in the limit; a handful of sweeps for a few
/// constraints). Stops when a full sweep moves the point by less than `tol`
/// in every coordinate, or after `max_sweeps` (an infeasible intersection
/// never converges; the caller checks feasibility afterwards).
fn project_onto_polytope(u: &mut [f64], ubox: &[(f64, f64)], half_spaces: &[HalfSpace], tol: f64, max_sweeps: usize) {
    let n = u.len();
    let sets = 1 + half_spaces.len();
    let mut increments = vec![vec![0.0; n]; sets];
    let mut y = vec![0.0; n];
    for _ in 0..max_sweeps {
        let mut moved: f64 = 0.0;
        for k in 0..sets {
            for i in 0..n {
                y[i] = u[i] + increments[k][i];
            }
            if k == 0 {
                for i in 0..n {
                    let (lo, hi) = ubox[i];
                    let p = y[i].max(lo).min(hi);
                    increments[k][i] = y[i] - p;
                    moved = moved.max((p - u[i]).abs());
                    u[i] = p;
                }
            } else {
                let h = &half_spaces[k - 1];
                let v = h.a.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>() - h.b;
                let step = if v < 0.0 && h.aa > 0.0 { -v / h.aa } else { 0.0 };
                for i in 0..n {
                    let p = y[i] + step * h.a[i];
                    increments[k][i] = y[i] - p;
                    moved = moved.max((p - u[i]).abs());
                    u[i] = p;
                }
            }
        }
        if moved <= tol {
            break;
        }
    }
}

/// Deterministic per-run seed: mixes the base seed with the bits of the
/// starting point (splitmix64 finaliser), so identical starts give identical
/// CMA-ES runs regardless of thread scheduling.
fn cmaes_seed(base: u64, x0: &[f64]) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    let mut h = mix(base ^ 0x9e37_79b9_7f4a_7c15);
    for &xi in x0 {
        h = mix(h ^ xi.to_bits().wrapping_add(0x9e37_79b9_7f4a_7c15));
    }
    h
}

/// Run one CMA-ES local minimization via the `cmaes` crate.
///
/// The starting point is evaluated first and kept as the incumbent, so the
/// result is never worse than `x0` (matching the NLopt algorithms, which all
/// evaluate `x0`). Termination on any of the crate's tolerance criteria or on
/// the evaluation/time budget counts as success; numerical breakdowns
/// (`InvalidFunctionValue`, `PosDefCov`, `TolXUp`) are reported as failures
/// but still return the best point seen.
///
/// Constraints: every sample is mirrored across any `linear` constraint it
/// violates (and folded into the box) before evaluation, so those hold
/// exactly; Dykstra projection onto the polytope is the fallback where
/// mirroring does not settle. Nonlinear `constraints` are handled by a feasibility-first
/// penalty: an infeasible sample is not evaluated and receives a value above
/// the start point's, increasing with its violation, so CMA-ES's ranking
/// pushes the population back into the feasible region. The returned point
/// is always feasible; if none was found the result is a failure.
fn run_cmaes(
    func: ScalarFn<'_>,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: &[ScalarFn<'_>],
    linear: &[LinearConstraint],
    options: &LocalOptimizerOptions,
) -> LocalOptResult {
    let dim = x0.len();
    if dim == 0 {
        return LocalOptResult::failure(x0, "CMA-ES needs at least one dimension".to_string());
    }
    let cfg = &options.cmaes;
    if !(cfg.sigma0.is_finite() && cfg.sigma0 > 0.0) {
        return LocalOptResult::failure(
            x0,
            format!("Invalid CMA-ES sigma0 {} (must be finite and > 0)", cfg.sigma0),
        );
    }

    let map = BoxMap::new(bounds);
    let ubox = map.u_bounds();
    let half_spaces = map.half_spaces(linear);
    let ctol = options.constraint_tol.max(0.0);
    let disp = options.disp;
    let fev_count = AtomicUsize::new(0);
    // Evaluate through the box map; counts every evaluation.
    let eval_box = |x: &[f64]| -> f64 {
        let n = fev_count.fetch_add(1, Ordering::Relaxed) + 1;
        let val = func(x);
        if disp {
            println!("  [Cmaes eval #{:>3}] f = {:+.6e}", n, val);
        }
        val
    };
    // Search coordinates -> feasible box point. Out-of-box coordinates are
    // folded back (triangle wave) and a sample beyond a linear constraint is
    // mirrored across it, so the landscape CMA-ES sees has no flat repaired
    // region (projection would map everything outside onto the boundary and
    // let the mean drift out, stalling the search on the boundary). A few
    // alternating rounds settle interactions between constraints and the
    // box; Dykstra projection is the fallback for a corner where mirroring
    // keeps bouncing.
    let to_feasible = |z: &[f64]| -> Vec<f64> {
        let mut u = map.fold(z);
        if !half_spaces.is_empty() {
            let mut settled = false;
            for _ in 0..16 {
                let mut changed = false;
                for h in &half_spaces {
                    if h.aa == 0.0 {
                        continue;
                    }
                    let v = h.a.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>() - h.b;
                    if v < 0.0 {
                        let step = -2.0 * v / h.aa;
                        for (ui, ai) in u.iter_mut().zip(&h.a) {
                            *ui += step * ai;
                        }
                        changed = true;
                    }
                }
                if !changed {
                    settled = true;
                    break;
                }
                u = map.fold(&u);
            }
            if !settled {
                project_onto_polytope(&mut u, &ubox, &half_spaces, 1e-12, 500);
            }
        }
        map.unscale(&u)
    };
    // Total constraint violation at a box point (0 when feasible). Linear
    // constraints are included for the case of an infeasible polytope.
    let violation = |x: &[f64]| -> f64 {
        let mut v = 0.0;
        for g in constraints {
            let gv = g(x);
            if gv.is_nan() {
                v += 1.0;
            } else if gv < -ctol {
                v -= gv;
            }
        }
        for c in linear {
            let gv = c.value(x);
            if gv < -ctol {
                v -= gv;
            }
        }
        v
    };

    // Incumbent: the starting point itself (projected onto the linear
    // constraints if it violates them).
    let mut best_x = to_feasible(&map.to_internal(x0));
    let mut best_f = if violation(&best_x) == 0.0 {
        eval_box(&best_x)
    } else {
        f64::INFINITY
    };
    // Infeasible samples rank above the start point by their violation.
    let penalty_base = if best_f.is_finite() {
        best_f + best_f.abs().max(1.0)
    } else {
        1.0
    };

    let reserved = 1 + usize::from(cfg.eval_final_mean);
    let z0 = map.to_internal(&best_x);
    let mean = DVector::from_vec(z0);
    let tol_fun = options.ftol_abs.max(0.0);
    let tol_x = options
        .xtol_rel
        .max(options.xtol_abs / map.min_scale())
        .max(0.0);

    let mut builder = CMAESOptions::new(mean, cfg.sigma0)
        .seed(cmaes_seed(cfg.seed, x0))
        .tol_fun(tol_fun)
        .tol_fun_hist(tol_fun)
        .tol_x(tol_x);
    if let Some(lambda) = cfg.population_size {
        builder = builder.population_size(lambda);
    }
    if let Some(maxeval) = options.maxeval {
        builder = builder.max_function_evals((maxeval as usize).saturating_sub(reserved).max(1));
    }
    if let Some(maxtime) = options.maxtime {
        if maxtime.is_finite() && maxtime > 0.0 {
            builder = builder.max_time(std::time::Duration::from_secs_f64(maxtime));
        }
    }

    let objective = |z: &DVector<f64>| -> f64 {
        let x = to_feasible(z.as_slice());
        let v = violation(&x);
        if v > 0.0 {
            return penalty_base + v;
        }
        eval_box(&x)
    };
    let mut state = match builder.build(&objective) {
        Ok(s) => s,
        Err(e) => {
            return LocalOptResult {
                x: best_x,
                fun: best_f,
                success: false,
                message: format!("Optimization failed: invalid CMA-ES options ({:?})", e),
                nfev: fev_count.load(Ordering::Relaxed),
                nit: 0,
            }
        }
    };
    let data = if cfg.parallel_eval {
        state.run_parallel()
    } else {
        state.run()
    };
    let generations = state.generation();

    // Best evaluated point, then optionally the final mean. Only feasible
    // points can be returned (a penalised sample never beats the incumbent
    // unless the start itself was infeasible, hence the explicit check).
    let better = |f: f64, incumbent: f64| f < incumbent || (incumbent.is_nan() && !f.is_nan());
    if let Some(ind) = data.overall_best.as_ref() {
        let x = to_feasible(ind.point.as_slice());
        if violation(&x) == 0.0 && better(ind.value, best_f) {
            best_x = x;
            best_f = ind.value;
        }
    }
    if cfg.eval_final_mean {
        let xm = to_feasible(data.final_mean.as_slice());
        if violation(&xm) == 0.0 {
            let fm = eval_box(&xm);
            if better(fm, best_f) {
                best_x = xm;
                best_f = fm;
            }
        }
    }
    let final_fev = fev_count.load(Ordering::Relaxed);

    let broke_down = data.reasons.iter().any(|r| {
        matches!(
            r,
            TerminationReason::InvalidFunctionValue
                | TerminationReason::PosDefCov
                | TerminationReason::TolXUp
        )
    });
    let reasons: Vec<String> = data.reasons.iter().map(|r| r.to_string()).collect();
    let reasons = reasons.join(", ");
    if disp {
        println!(
            "  [Cmaes] {}: f_best = {:+.6e} ({} evals, {} generations)",
            reasons, best_f, final_fev, generations
        );
    }

    if !best_f.is_finite() {
        let why = if constraints.is_empty() && linear.is_empty() {
            "objective is non-finite at the returned point"
        } else {
            "no feasible point with a finite objective was found"
        };
        LocalOptResult {
            x: best_x,
            fun: best_f,
            success: false,
            message: format!("Optimization failed: {} ({})", why, reasons),
            nfev: final_fev,
            nit: generations,
        }
    } else if broke_down {
        LocalOptResult {
            x: best_x,
            fun: best_f,
            success: false,
            message: format!("Optimization failed: CMA-ES terminated on {}", reasons),
            nfev: final_fev,
            nit: generations,
        }
    } else {
        LocalOptResult {
            x: best_x,
            fun: best_f,
            success: true,
            message: format!("Optimization succeeded: CMA-ES terminated on {}", reasons),
            nfev: final_fev,
            nit: generations,
        }
    }
}

/// Perform local minimization using NLOPT (or the `cmaes` crate for
/// `LocalOptimizer::Cmaes`).
///
/// # Arguments
///
/// * `func` - The objective function to minimize.
/// * `x0` - Initial guess for the optimization parameters.
/// * `bounds` - Bounds for each dimension as (lower, upper) pairs.
/// * `constraints` - Optional inequality constraints where g(x) >= 0 means
///   feasible. SHGO uses g(x) >= 0 convention, we convert to NLOPT's g(x) <= 0.
///   If the chosen algorithm cannot handle constraints the run is upgraded to
///   COBYLA.
/// * `options` - Local optimizer configuration.
///
/// # Returns
///
/// Returns a `LocalOptResult` with the optimization results.
///
/// # Note
///
/// Since NLOPT's `Nlopt` struct is `!Send` and `!Sync`, this function creates
/// a fresh optimizer instance for each call. For parallel optimization, each
/// thread should call this function independently. The objective must be
/// `Sync` because gradient-based algorithms evaluate their finite-difference
/// gradients in parallel.
pub fn minimize_local<F, G>(
    func: &F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: Option<&[G]>,
    options: &LocalOptimizerOptions,
) -> LocalOptResult
where
    F: Fn(&[f64]) -> f64 + Sync,
    G: Fn(&[f64]) -> f64 + Sync,
{
    let cons: Vec<ScalarFn<'_>> = constraints
        .map(|cs| cs.iter().map(|g| g as ScalarFn<'_>).collect())
        .unwrap_or_default();
    run_local(func, x0, bounds, &cons, &[], options)
}

/// Perform local minimization with constraints.
///
/// This version adds inequality constraints to the optimizer for algorithms
/// that support them (COBYLA, SLSQP); other algorithms are upgraded to COBYLA
/// or wrapped, per `options.constraint_handling`. See
/// [`minimize_local_with_constraints`] to pass linear constraints as well.
///
/// # Arguments
///
/// * `func` - The objective function to minimize.
/// * `x0` - Initial guess for the optimization parameters.
/// * `bounds` - Bounds for each dimension as (lower, upper) pairs.
/// * `constraints` - Inequality constraints where g(x) >= 0 means feasible.
/// * `options` - Local optimizer configuration.
///
/// # Returns
///
/// Returns a `LocalOptResult` with the optimization results.
pub fn minimize_local_constrained<F>(
    func: F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: &[BoxedConstraint],
    options: &LocalOptimizerOptions,
) -> LocalOptResult
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    minimize_local_with_constraints(func, x0, bounds, constraints, &[], options)
}

/// Perform local minimization with nonlinear (`g(x) >= 0` closures) and
/// linear ([`LinearConstraint`]) inequality constraints.
///
/// COBYLA and SLSQP take both kinds natively; CMA-ES takes linear constraints
/// natively by projecting every sample onto them. Any other combination is
/// resolved by `options.constraint_handling` (see [`ConstraintHandling`]).
pub fn minimize_local_with_constraints<F>(
    func: F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: &[BoxedConstraint],
    linear: &[LinearConstraint],
    options: &LocalOptimizerOptions,
) -> LocalOptResult
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let cons: Vec<ScalarFn<'_>> = constraints.iter().map(|b| &**b as ScalarFn<'_>).collect();
    run_local(&func, x0, bounds, &cons, linear, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Sphere function: f(x) = sum(x_i^2)
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi.powi(2)).sum()
    }

    // Rosenbrock function
    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    }

    #[test]
    fn test_local_optimizer_default() {
        let opts = LocalOptimizerOptions::default();
        assert_eq!(opts.algorithm, LocalOptimizer::Bobyqa);
        assert_eq!(opts.ftol_rel, 1e-8);
        assert_eq!(opts.maxeval, Some(1000));
    }

    #[test]
    fn test_algorithm_conversion() {
        assert!(matches!(
            LocalOptimizer::Bobyqa.to_nlopt_algorithm(),
            Some(Algorithm::Bobyqa)
        ));
        assert!(matches!(
            LocalOptimizer::Cobyla.to_nlopt_algorithm(),
            Some(Algorithm::Cobyla)
        ));
        assert!(matches!(
            LocalOptimizer::Slsqp.to_nlopt_algorithm(),
            Some(Algorithm::Slsqp)
        ));
        assert!(LocalOptimizer::Cmaes.to_nlopt_algorithm().is_none());
        assert!(!LocalOptimizer::Cmaes.is_nlopt());
        assert!(LocalOptimizer::Sbplx.is_nlopt());
    }

    #[test]
    fn test_algorithm_properties() {
        assert!(LocalOptimizer::Cobyla.supports_constraints());
        assert!(LocalOptimizer::Slsqp.supports_constraints());
        assert!(!LocalOptimizer::Bobyqa.supports_constraints());

        assert!(LocalOptimizer::Slsqp.requires_gradient());
        assert!(LocalOptimizer::Lbfgs.requires_gradient());
        assert!(!LocalOptimizer::Bobyqa.requires_gradient());
        assert!(!LocalOptimizer::Cmaes.supports_constraints());
        assert!(!LocalOptimizer::Cmaes.requires_gradient());
    }

    #[test]
    fn test_fd_gradient_matches_analytic() {
        let x = [0.7, -1.3, 2.0];
        let bounds = [(-5.0, 5.0); 3];
        let mut g = [0.0; 3];
        let n = fd_gradient(&sphere, &x, sphere(&x), &bounds, &mut g);
        assert_eq!(n, 3);
        for (gi, xi) in g.iter().zip(x.iter()) {
            assert_relative_eq!(*gi, 2.0 * xi, epsilon = 1e-5);
        }
        // At the upper bound the step must go backwards and stay in bounds.
        let x = [5.0, 0.0];
        let f = |v: &[f64]| {
            assert!(v[0] <= 5.0, "finite-difference probe left the bounds");
            sphere(v)
        };
        let mut g = [0.0; 2];
        fd_gradient(&f, &x, sphere(&x), &[(-5.0, 5.0); 2], &mut g);
        assert_relative_eq!(g[0], 10.0, epsilon = 1e-5);
    }

    #[test]
    fn test_minimize_sphere_bobyqa() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];
        let options = LocalOptimizerOptions::default();

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_minimize_sphere_neldermead() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::NelderMead,
            ..Default::default()
        };

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_minimize_sphere_cobyla() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cobyla,
            ..Default::default()
        };

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_minimize_sphere_praxis() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Praxis,
            ..Default::default()
        };

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
    }

    /// Gradient-based algorithms must actually move (regression: they used to
    /// receive no gradient and returned the starting point after one evaluation).
    #[test]
    fn test_minimize_gradient_based_moves() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];
        for alg in [LocalOptimizer::Slsqp, LocalOptimizer::Lbfgs] {
            let options = LocalOptimizerOptions {
                algorithm: alg,
                ..Default::default()
            };
            let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
            assert!(result.success, "{:?} failed: {}", alg, result.message);
            assert!(result.nfev > 1, "{:?} made only {} evaluation(s)", alg, result.nfev);
            assert_relative_eq!(result.fun, 0.0, epsilon = 1e-8);
            assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-4);
            assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-4);

            let result = minimize_local(&rosenbrock, &[-1.0, 2.0], &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
            assert!(result.success, "{:?} failed: {}", alg, result.message);
            assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
            assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_minimize_rosenbrock() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![0.0, 0.0];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Bobyqa,
            maxeval: Some(5000),
            ..Default::default()
        };

        let result = minimize_local(&rosenbrock, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(result.success);
        // Rosenbrock minimum is at (1, 1) with value 0
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_minimize_with_maxeval() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![4.0, 4.0]; // Start far from minimum
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Bobyqa,
            maxeval: Some(10), // Very few evaluations
            ..Default::default()
        };

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        // Should terminate due to maxeval, may or may not succeed
        assert!(result.nfev <= 15); // Allow some slack
    }

    #[test]
    fn test_minimize_dimension_mismatch() {
        let bounds = vec![(-5.0, 5.0)]; // 1D bounds
        let x0 = vec![1.0, 1.0]; // 2D starting point
        let options = LocalOptimizerOptions::default();

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(!result.success);
        assert!(result.message.contains("Dimension mismatch"));
    }

    #[test]
    fn test_minimize_different_starting_points() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let options = LocalOptimizerOptions::default();

        // Test from different starting points
        for x0 in [
            vec![-4.0, -4.0],
            vec![4.0, 4.0],
            vec![-4.0, 4.0],
            vec![4.0, -4.0],
        ] {
            let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
            assert!(result.success, "Failed from starting point {:?}", x0);
            assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_minimize_higher_dimension() {
        let dim = 5;
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); dim];
        let x0: Vec<f64> = vec![1.0; dim];
        let options = LocalOptimizerOptions {
            maxeval: Some(2000),
            ..Default::default()
        };

        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);

        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-5);
        for xi in &result.x {
            assert_relative_eq!(*xi, 0.0, epsilon = 1e-3);
        }
    }

    /// Constraints passed to the public `minimize_local` must be honoured
    /// (regression: they used to be silently dropped, returning the
    /// unconstrained minimum).
    #[test]
    fn test_minimize_constrained_fallback() {
        // When using BOBYQA with constraints, should fall back to COBYLA
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let x0 = vec![1.5, 1.5];

        // Constraint: x[0] + x[1] >= 1  -> constrained optimum (0.5, 0.5), f = 0.5
        let constraints = [|x: &[f64]| x[0] + x[1] - 1.0];

        for alg in [LocalOptimizer::Bobyqa, LocalOptimizer::Cobyla, LocalOptimizer::Slsqp] {
            let options = LocalOptimizerOptions {
                algorithm: alg,
                ..Default::default()
            };
            let result = minimize_local(&sphere, &x0, &bounds, Some(&constraints[..]), &options);
            assert!(result.success, "{:?}: {}", alg, result.message);
            let g = result.x[0] + result.x[1] - 1.0;
            assert!(g >= -1e-6, "{:?} violated the constraint: g = {}", alg, g);
            assert_relative_eq!(result.fun, 0.5, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_minimize_local_constrained_boxed() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let constraints: Vec<BoxedConstraint> = vec![Box::new(|x: &[f64]| x[0] + x[1] - 1.0)];
        for alg in [LocalOptimizer::Cobyla, LocalOptimizer::Slsqp] {
            let options = LocalOptimizerOptions {
                algorithm: alg,
                ..Default::default()
            };
            let result = minimize_local_constrained(sphere, &[1.5, 1.5], &bounds, &constraints, &options);
            assert!(result.success, "{:?}: {}", alg, result.message);
            assert!(result.x[0] + result.x[1] - 1.0 >= -1e-6, "{:?} violated the constraint", alg);
            assert_relative_eq!(result.fun, 0.5, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_all_algorithms_on_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];

        let algorithms = [
            LocalOptimizer::Bobyqa,
            LocalOptimizer::Cobyla,
            LocalOptimizer::Slsqp,
            LocalOptimizer::Lbfgs,
            LocalOptimizer::NelderMead,
            LocalOptimizer::Praxis,
            LocalOptimizer::NewuoaBound,
            LocalOptimizer::Sbplx,
            LocalOptimizer::Cmaes,
        ];

        for alg in algorithms {
            let options = LocalOptimizerOptions {
                algorithm: alg,
                maxeval: Some(2000),
                ..Default::default()
            };

            let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
            assert!(
                result.success,
                "Algorithm {:?} failed: {}",
                alg,
                result.message
            );
            assert_relative_eq!(result.fun, 0.0, epsilon = 1e-5);
        }
    }
    #[test]
    fn test_reflect_into_box_properties() {
        // Identity inside the box, bounds reachable exactly, fold-back outside.
        assert_eq!(reflect_into(0.3, 0.0, 1.0), 0.3);
        assert_eq!(reflect_into(0.0, 0.0, 1.0), 0.0);
        assert_eq!(reflect_into(1.0, 0.0, 1.0), 1.0);
        assert_relative_eq!(reflect_into(1.25, 0.0, 1.0), 0.75);
        assert_relative_eq!(reflect_into(-0.25, 0.0, 1.0), 0.25);
        assert_relative_eq!(reflect_into(2.25, 0.0, 1.0), 0.25); // period 2
        assert_relative_eq!(reflect_into(-3.0, -5.0, 5.0), -3.0);
        assert_relative_eq!(reflect_into(7.0, -5.0, 5.0), 3.0);
        // One-sided and unbounded coordinates.
        assert_relative_eq!(reflect_into(-2.0, 0.0, f64::INFINITY), 2.0);
        assert_relative_eq!(reflect_into(3.0, f64::NEG_INFINITY, 1.0), -1.0);
        assert_eq!(reflect_into(42.0, f64::NEG_INFINITY, f64::INFINITY), 42.0);
        // Degenerate box collapses to the single feasible value.
        assert_eq!(reflect_into(9.0, 2.0, 2.0), 2.0);
        // Continuity across many periods: never leaves the box.
        for i in -200..200 {
            let v = i as f64 * 0.37;
            let r = reflect_into(v, -1.5, 2.5);
            assert!((-1.5..=2.5).contains(&r), "{} -> {}", v, r);
        }
    }

    #[test]
    fn test_box_map_round_trip() {
        let bounds = [(-5.0, 5.0), (0.0, 1.0), (f64::NEG_INFINITY, f64::INFINITY)];
        let map = BoxMap::new(&bounds);
        let x = [2.5, 0.25, -7.0];
        let z = map.to_internal(&x);
        assert_relative_eq!(z[0], 0.75);
        assert_relative_eq!(z[1], 0.25);
        assert_relative_eq!(z[2], -7.0);
        let back = map.unscale(&map.fold(&z));
        for (a, b) in back.iter().zip(&x) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        assert_relative_eq!(map.min_scale(), 1.0);
    }

    #[test]
    fn test_minimize_sphere_cmaes() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            ..Default::default()
        };
        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
        assert!(result.success, "{}", result.message);
        assert!(result.message.contains("CMA-ES"));
        assert!(result.nit > 0, "generations should be reported in nit");
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    /// The unconstrained minimum lies outside the box: the answer must sit on
    /// the boundary and no probe may ever leave the bounds.
    #[test]
    fn test_cmaes_respects_bounds() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let shifted = |x: &[f64]| {
            for (i, xi) in x.iter().enumerate() {
                assert!(
                    *xi >= bounds[i].0 && *xi <= bounds[i].1,
                    "CMA-ES evaluated an out-of-bounds point {:?}",
                    x
                );
            }
            (x[0] - 7.0).powi(2) + (x[1] - 7.0).powi(2)
        };
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            maxeval: Some(4000),
            ..Default::default()
        };
        let result = minimize_local(&shifted, &[0.0, 0.0], &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
        assert!(result.success, "{}", result.message);
        assert_relative_eq!(result.x[0], 5.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 5.0, epsilon = 1e-4);
        assert_relative_eq!(result.fun, 8.0, epsilon = 1e-3);
    }

    #[test]
    fn test_cmaes_is_deterministic() {
        let bounds = vec![(-5.0, 5.0); 3];
        let x0 = vec![2.0, -1.0, 0.5];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            ..Default::default()
        };
        let a = minimize_local(&rosenbrock3, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
        let b = minimize_local(&rosenbrock3, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
        assert_eq!(a.x, b.x);
        assert_eq!(a.fun.to_bits(), b.fun.to_bits());
        assert_eq!(a.nfev, b.nfev);
        // A different base seed gives a different (but still converged) run.
        let options2 = LocalOptimizerOptions {
            cmaes: CmaesOptions { seed: 7, ..Default::default() },
            ..options.clone()
        };
        let c = minimize_local(&rosenbrock3, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options2);
        assert!(c.success, "{}", c.message);
        assert!(a.nfev != c.nfev || a.x != c.x, "seed had no effect");
    }

    fn rosenbrock3(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    /// The start is evaluated and kept as the incumbent, so even a starved
    /// budget returns something no worse than `x0`.
    #[test]
    fn test_cmaes_never_worse_than_start() {
        let bounds = vec![(-5.0, 5.0); 4];
        let x0 = vec![0.01, -0.02, 0.005, 0.0];
        let f0 = sphere(&x0);
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            maxeval: Some(3),
            cmaes: CmaesOptions { sigma0: 0.4, ..Default::default() },
            ..Default::default()
        };
        let result = minimize_local(&sphere, &x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
        assert!(result.fun <= f0, "{} > f(x0) = {}", result.fun, f0);
        assert!(result.nfev >= 1);
        assert!(result.nfev <= 3 + 12, "budget overshoot: {}", result.nfev);
    }

    #[test]
    fn test_cmaes_unbounded_coordinates() {
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)];
        let f = |x: &[f64]| {
            assert!(x[1] >= 0.0, "left the one-sided bound: {:?}", x);
            (x[0] - 1.5).powi(2) + (x[1] + 2.0).powi(2)
        };
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            ..Default::default()
        };
        let result = minimize_local(&f, &[0.0, 1.0], &bounds, None::<&[fn(&[f64]) -> f64]>, &options);
        assert!(result.success, "{}", result.message);
        assert_relative_eq!(result.x[0], 1.5, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_cmaes_rejects_bad_sigma() {
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            cmaes: CmaesOptions { sigma0: 0.0, ..Default::default() },
            ..Default::default()
        };
        let result = minimize_local(&sphere, &[1.0, 1.0], &[(-5.0, 5.0); 2], None::<&[fn(&[f64]) -> f64]>, &options);
        assert!(!result.success);
        assert!(result.message.contains("sigma0"));
        assert_eq!(result.nfev, 0);
    }

    /// A bowl with small high-frequency ripples (see
    /// `examples/local_optimizer_benchmark.rs`): the quadratic-model method
    /// stops in the nearest ripple, the population method with an enlarged
    /// population reaches the bottom. Pins the behaviour the CMA-ES backend
    /// exists for; the seeds make it deterministic.
    #[test]
    fn test_cmaes_escapes_ripples_where_bobyqa_stalls() {
        let c = [0.3, -0.4];
        let rugged = |x: &[f64]| -> f64 {
            x.iter()
                .zip(&c)
                .map(|(a, b)| {
                    let d = a - b;
                    d * d + 0.15 * (1.0 - (2.0 * std::f64::consts::PI * 3.0 * d).cos())
                })
                .sum()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let starts = [[2.5, 2.5], [-3.0, 1.0], [1.0, -3.0], [4.0, -4.0]];

        let bobyqa = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Bobyqa,
            maxeval: Some(4000),
            ..Default::default()
        };
        let cmaes = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            maxeval: Some(4000),
            cmaes: CmaesOptions { population_size: Some(32), ..Default::default() },
            ..Default::default()
        };
        let mut bobyqa_stalled = 0;
        for x0 in &starts {
            let b = minimize_local(&rugged, x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &bobyqa);
            if b.fun > 1e-3 {
                bobyqa_stalled += 1;
            }
            let c = minimize_local(&rugged, x0, &bounds, None::<&[fn(&[f64]) -> f64]>, &cmaes);
            assert!(c.success, "{}", c.message);
            assert!(c.fun < 1e-6, "CMA-ES from {:?} ended at f = {} ({})", x0, c.fun, c.message);
        }
        assert!(
            bobyqa_stalled >= 2,
            "BOBYQA stalled from only {} of {} starts; the ripples are no longer a trap",
            bobyqa_stalled,
            starts.len()
        );
    }

    /// With constraints CMA-ES is upgraded to COBYLA like the other
    /// bound-only algorithms.
    #[test]
    fn test_cmaes_constrained_upgrades_to_cobyla() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let constraints = [|x: &[f64]| x[0] + x[1] - 1.0];
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            ..Default::default()
        };
        let result = minimize_local(&sphere, &[1.5, 1.5], &bounds, Some(&constraints[..]), &options);
        assert!(result.success, "{}", result.message);
        assert!(!result.message.contains("CMA-ES"));
        assert!(result.x[0] + result.x[1] - 1.0 >= -1e-6);
        assert_relative_eq!(result.fun, 0.5, epsilon = 1e-4);
    }
    #[test]
    fn test_linear_constraint_forms() {
        let ge = LinearConstraint::ge(vec![1.0, 1.0], 1.0); // x0 + x1 >= 1
        assert_relative_eq!(ge.value(&[0.75, 0.75]), 0.5);
        assert!(ge.is_satisfied(&[0.5, 0.5], 0.0));
        assert!(!ge.is_satisfied(&[0.2, 0.2], 1e-9));
        let le = LinearConstraint::le(vec![1.0, 1.0], 1.0); // x0 + x1 <= 1
        assert_eq!(le.a, vec![-1.0, -1.0]);
        assert_eq!(le.b, -1.0);
        assert!(le.is_satisfied(&[0.2, 0.2], 0.0));
        assert!(!le.is_satisfied(&[0.75, 0.75], 0.0));
    }

    #[test]
    fn test_effective_algorithm_rules() {
        let with = |algorithm, constraint_handling| LocalOptimizerOptions {
            algorithm,
            constraint_handling,
            ..Default::default()
        };
        use ConstraintHandling::{KeepAlgorithm, UpgradeToCobyla};
        use LocalOptimizer::*;
        // Nothing to handle: always the chosen algorithm.
        assert_eq!(effective_algorithm(&with(Bobyqa, UpgradeToCobyla), false, false), Bobyqa);
        // Native support is never overridden.
        assert_eq!(effective_algorithm(&with(Slsqp, UpgradeToCobyla), true, true), Slsqp);
        assert_eq!(effective_algorithm(&with(Cobyla, KeepAlgorithm), true, true), Cobyla);
        // Bound-only NLopt methods: upgrade by default, keep on request.
        assert_eq!(effective_algorithm(&with(Bobyqa, UpgradeToCobyla), true, false), Cobyla);
        assert_eq!(effective_algorithm(&with(Bobyqa, UpgradeToCobyla), false, true), Cobyla);
        assert_eq!(effective_algorithm(&with(Sbplx, KeepAlgorithm), true, false), Sbplx);
        // CMA-ES: linear constraints are native, nonlinear ones follow the setting.
        assert_eq!(effective_algorithm(&with(Cmaes, UpgradeToCobyla), false, true), Cmaes);
        assert_eq!(effective_algorithm(&with(Cmaes, UpgradeToCobyla), true, false), Cobyla);
        assert_eq!(effective_algorithm(&with(Cmaes, KeepAlgorithm), true, true), Cmaes);
    }

    #[test]
    fn test_polytope_projection() {
        let ubox = [(0.0, 1.0), (0.0, 1.0)];
        // u0 + u1 <= 1  ->  -u0 - u1 >= -1
        let hs = [HalfSpace { a: vec![-1.0, -1.0], b: -1.0, aa: 2.0 }];
        // Feasible point is unchanged.
        let mut u = vec![0.2, 0.1];
        project_onto_polytope(&mut u, &ubox, &hs, 1e-12, 500);
        assert_relative_eq!(u[0], 0.2, epsilon = 1e-10);
        assert_relative_eq!(u[1], 0.1, epsilon = 1e-10);
        // Onto the half-space.
        let mut u = vec![1.0, 1.0];
        project_onto_polytope(&mut u, &ubox, &hs, 1e-12, 500);
        assert_relative_eq!(u[0], 0.5, epsilon = 1e-9);
        assert_relative_eq!(u[1], 0.5, epsilon = 1e-9);
        // Onto a vertex of box ∩ half-space: nearest feasible point to
        // (1.5, -0.2) in the triangle (0,0),(1,0),(0,1) is the corner (1,0).
        let mut u = vec![1.5, -0.2];
        project_onto_polytope(&mut u, &ubox, &hs, 1e-12, 500);
        assert_relative_eq!(u[0], 1.0, epsilon = 1e-9);
        assert_relative_eq!(u[1], 0.0, epsilon = 1e-9);
        // Two half-spaces: u0 >= 0.6 and u0 + u1 <= 1, from (0.2, 0.9) ->
        // the projection onto the segment u0 = 0.6, u1 <= 0.4 is (0.6, 0.4).
        let hs2 = [
            HalfSpace { a: vec![1.0, 0.0], b: 0.6, aa: 1.0 },
            HalfSpace { a: vec![-1.0, -1.0], b: -1.0, aa: 2.0 },
        ];
        let mut u = vec![0.2, 0.9];
        project_onto_polytope(&mut u, &ubox, &hs2, 1e-12, 500);
        assert_relative_eq!(u[0], 0.6, epsilon = 1e-9);
        assert_relative_eq!(u[1], 0.4, epsilon = 1e-9);
    }

    /// CMA-ES with a linear constraint stays CMA-ES (no COBYLA upgrade) and
    /// never evaluates an infeasible point.
    #[test]
    fn test_cmaes_mirrors_across_linear_constraint() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let linear = [LinearConstraint::ge(vec![1.0, 1.0], 1.0)]; // optimum (0.5, 0.5), f = 0.5
        let checked = |x: &[f64]| {
            assert!(x[0] + x[1] - 1.0 >= -1e-9, "evaluated an infeasible point {:?}", x);
            sphere(x)
        };
        for handling in [ConstraintHandling::UpgradeToCobyla, ConstraintHandling::KeepAlgorithm] {
            let options = LocalOptimizerOptions {
                algorithm: LocalOptimizer::Cmaes,
                constraint_handling: handling,
                ..Default::default()
            };
            let r = minimize_local_with_constraints(checked, &[1.5, 1.5], &bounds, &[], &linear, &options);
            assert!(r.success, "{:?}: {}", handling, r.message);
            assert!(r.message.contains("CMA-ES"), "{:?}: {}", handling, r.message);
            assert_relative_eq!(r.fun, 0.5, epsilon = 1e-6);
            assert_relative_eq!(r.x[0], 0.5, epsilon = 1e-3);
            assert_relative_eq!(r.x[1], 0.5, epsilon = 1e-3);
        }
        // A start outside the polytope is projected in first.
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            ..Default::default()
        };
        let r = minimize_local_with_constraints(checked, &[0.1, 0.1], &bounds, &[], &linear, &options);
        assert!(r.success, "{}", r.message);
        assert_relative_eq!(r.fun, 0.5, epsilon = 1e-6);
    }

    /// Nonlinear closure constraints under `KeepAlgorithm`: the penalty path
    /// keeps every objective evaluation feasible and lands on the boundary.
    #[test]
    fn test_cmaes_penalty_for_nonlinear_constraints() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let constraints: Vec<BoxedConstraint> = vec![Box::new(|x: &[f64]| x[0] + x[1] - 1.0)];
        let checked = |x: &[f64]| {
            assert!(x[0] + x[1] - 1.0 >= -1e-8, "evaluated an infeasible point {:?}", x);
            sphere(x)
        };
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Cmaes,
            constraint_handling: ConstraintHandling::KeepAlgorithm,
            maxeval: Some(4000),
            ..Default::default()
        };
        let r = minimize_local_constrained(checked, &[1.5, 1.5], &bounds, &constraints, &options);
        assert!(r.success, "{}", r.message);
        assert!(r.message.contains("CMA-ES"), "{}", r.message);
        assert!(r.x[0] + r.x[1] - 1.0 >= -1e-8);
        assert_relative_eq!(r.fun, 0.5, epsilon = 1e-3);
        // Infeasible start and nothing feasible reachable: reported as failure.
        let impossible: Vec<BoxedConstraint> = vec![Box::new(|_: &[f64]| -1.0)];
        let r = minimize_local_constrained(sphere, &[1.0, 1.0], &bounds, &impossible, &options);
        assert!(!r.success);
        assert!(r.message.contains("no feasible point"), "{}", r.message);
        assert_eq!(r.nfev, 0, "the objective must not be evaluated at infeasible points");
    }

    /// `KeepAlgorithm` wraps bound-only NLopt methods in AUGLAG instead of
    /// switching to COBYLA.
    #[test]
    fn test_auglag_keeps_algorithm() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let constraints: Vec<BoxedConstraint> = vec![Box::new(|x: &[f64]| x[0] + x[1] - 1.0)];
        let linear = [LinearConstraint::ge(vec![1.0, 1.0], 1.0)];
        for alg in [LocalOptimizer::Bobyqa, LocalOptimizer::Sbplx, LocalOptimizer::NelderMead] {
            let options = LocalOptimizerOptions {
                algorithm: alg,
                constraint_handling: ConstraintHandling::KeepAlgorithm,
                maxeval: Some(5000),
                ..Default::default()
            };
            // Closure constraint.
            let r = minimize_local_constrained(sphere, &[1.5, 1.5], &bounds, &constraints, &options);
            assert!(r.success, "{:?}: {}", alg, r.message);
            assert!(r.message.contains("AUGLAG"), "{:?}: {}", alg, r.message);
            assert!(r.x[0] + r.x[1] - 1.0 >= -1e-6, "{:?} violated the constraint: {:?}", alg, r.x);
            assert_relative_eq!(r.fun, 0.5, epsilon = 1e-4);
            // Linear constraint, same answer.
            let r = minimize_local_with_constraints(sphere, &[1.5, 1.5], &bounds, &[], &linear, &options);
            assert!(r.success, "{:?}: {}", alg, r.message);
            assert!(r.message.contains("AUGLAG"), "{:?}: {}", alg, r.message);
            assert_relative_eq!(r.fun, 0.5, epsilon = 1e-4);
        }
        // Default handling still upgrades to COBYLA.
        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Bobyqa,
            ..Default::default()
        };
        let r = minimize_local_constrained(sphere, &[1.5, 1.5], &bounds, &constraints, &options);
        assert!(r.message.contains("Cobyla") && !r.message.contains("AUGLAG"), "{}", r.message);
    }

    #[test]
    fn test_linear_constraints_all_native_algorithms() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let linear = [LinearConstraint::le(vec![-1.0, -1.0], -1.0)]; // x0 + x1 >= 1
        for alg in [LocalOptimizer::Bobyqa, LocalOptimizer::Cobyla, LocalOptimizer::Slsqp, LocalOptimizer::Cmaes] {
            let options = LocalOptimizerOptions {
                algorithm: alg,
                ..Default::default()
            };
            let r = minimize_local_with_constraints(sphere, &[1.5, 1.5], &bounds, &[], &linear, &options);
            assert!(r.success, "{:?}: {}", alg, r.message);
            assert!(r.x[0] + r.x[1] - 1.0 >= -1e-6, "{:?} violated the constraint", alg);
            assert_relative_eq!(r.fun, 0.5, epsilon = 1e-4);
        }
        // Wrong coefficient count is rejected up front.
        let bad = [LinearConstraint::ge(vec![1.0], 0.0)];
        let r = minimize_local_with_constraints(sphere, &[1.5, 1.5], &bounds, &[], &bad, &LocalOptimizerOptions::default());
        assert!(!r.success);
        assert!(r.message.contains("Dimension mismatch"));
    }
}
