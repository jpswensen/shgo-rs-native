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
/// - Use `NelderMead` for noisy or non-smooth functions
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
}

impl LocalOptimizer {
    /// Convert to NLOPT Algorithm enum.
    pub fn to_nlopt_algorithm(self) -> Algorithm {
        match self {
            LocalOptimizer::Bobyqa => Algorithm::Bobyqa,
            LocalOptimizer::Cobyla => Algorithm::Cobyla,
            LocalOptimizer::Slsqp => Algorithm::Slsqp,
            LocalOptimizer::Lbfgs => Algorithm::Lbfgs,
            LocalOptimizer::NelderMead => Algorithm::Neldermead,
            LocalOptimizer::Praxis => Algorithm::Praxis,
            LocalOptimizer::NewuoaBound => Algorithm::NewuoaBound,
            LocalOptimizer::Sbplx => Algorithm::Sbplx,
        }
    }

    /// Check if the algorithm supports nonlinear constraints.
    pub fn supports_constraints(self) -> bool {
        matches!(self, LocalOptimizer::Cobyla | LocalOptimizer::Slsqp)
    }

    /// Check if the algorithm requires gradients (supplied by finite differences).
    pub fn requires_gradient(self) -> bool {
        matches!(self, LocalOptimizer::Slsqp | LocalOptimizer::Lbfgs)
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
        }
    }
}

/// Boxed inequality-constraint function (`g(x) >= 0` means feasible).
pub type BoxedConstraint = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// A scalar function reference usable from the NLOPT callbacks (and from the
/// rayon workers that evaluate finite-difference gradients).
type ScalarFn<'a> = &'a (dyn Fn(&[f64]) -> f64 + Sync);

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

/// Run one NLOPT local minimization. Shared by the two public entry points.
///
/// `constraints` are inequality constraints in SHGO's `g(x) >= 0` convention;
/// if any are present and `options.algorithm` cannot handle them the run is
/// upgraded to COBYLA. Gradient-based algorithms receive forward-difference
/// gradients of the objective and of every constraint.
fn run_nlopt(
    func: ScalarFn<'_>,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: &[ScalarFn<'_>],
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

    // Constrained problems need a constraint-capable algorithm.
    let algo = if !constraints.is_empty() && !options.algorithm.supports_constraints() {
        LocalOptimizer::Cobyla
    } else {
        options.algorithm
    };
    let needs_grad = algo.requires_gradient();

    // Track function evaluations (finite-difference evaluations included).
    let fev_count = AtomicUsize::new(0);
    let disp = options.disp;

    // Objective wrapper for NLOPT.
    // Signature: (&[f64], Option<&mut [f64]>, &mut UserData) -> f64
    let objective = |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let n = fev_count.fetch_add(1, Ordering::Relaxed) + 1;
        let val = func(x);
        if disp {
            println!("  [{:?} eval #{:>3}] f = {:+.6e}", algo, n, val);
        }
        if let Some(g) = grad {
            if needs_grad {
                let used = fd_gradient(func, x, val, bounds, g);
                fev_count.fetch_add(used, Ordering::Relaxed);
            }
        }
        val
    };

    // Create NLOPT optimizer
    let mut opt = Nlopt::new(algo.to_nlopt_algorithm(), dim, objective, Target::Minimize, ());

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
                "  [{:?}] {:?}: f_best = {:+.6e} ({} evals)",
                algo, state, fval, final_fev
            ),
            Err((state, fval)) => println!(
                "  [{:?}] FAILED {:?}: f_best = {:+.6e} ({} evals)",
                algo, state, fval, final_fev
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
            message: format!("Optimization succeeded: {:?}", success_state),
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

/// Perform local minimization using NLOPT.
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
    run_nlopt(func, x0, bounds, &cons, options)
}

/// Perform local minimization with constraints using NLOPT's constraint support.
///
/// This version adds inequality constraints to the optimizer for algorithms
/// that support them (COBYLA, SLSQP); other algorithms are upgraded to COBYLA.
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
    let cons: Vec<ScalarFn<'_>> = constraints.iter().map(|b| &**b as ScalarFn<'_>).collect();
    run_nlopt(&func, x0, bounds, &cons, options)
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
            Algorithm::Bobyqa
        ));
        assert!(matches!(
            LocalOptimizer::Cobyla.to_nlopt_algorithm(),
            Algorithm::Cobyla
        ));
        assert!(matches!(
            LocalOptimizer::Slsqp.to_nlopt_algorithm(),
            Algorithm::Slsqp
        ));
    }

    #[test]
    fn test_algorithm_properties() {
        assert!(LocalOptimizer::Cobyla.supports_constraints());
        assert!(LocalOptimizer::Slsqp.supports_constraints());
        assert!(!LocalOptimizer::Bobyqa.supports_constraints());

        assert!(LocalOptimizer::Slsqp.requires_gradient());
        assert!(LocalOptimizer::Lbfgs.requires_gradient());
        assert!(!LocalOptimizer::Bobyqa.requires_gradient());
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
}
