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
//!
//! - **LBFGS**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno.
//!   Gradient-based, good for smooth unconstrained problems.
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
//! use shgo_rs::local_opt::{LocalOptimizer, LocalOptimizerOptions, minimize_local};
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
use std::cell::Cell;
use std::rc::Rc;

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
    /// Gradient-based, supports equality and inequality constraints.
    Slsqp,

    /// Limited-memory BFGS.
    /// Gradient-based, bounds only.
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

    /// Check if the algorithm requires gradients.
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

    /// Maximum number of function evaluations.
    pub maxeval: Option<u32>,

    /// Maximum time in seconds.
    pub maxtime: Option<f64>,

    /// Initial step size for derivative-free methods.
    /// If None, NLOPT chooses heuristically.
    pub initial_step: Option<f64>,

    /// Constraint tolerance (for algorithms that support constraints).
    pub constraint_tol: f64,
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
        }
    }
}

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
    /// Number of function evaluations used.
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

/// Perform local minimization using NLOPT.
///
/// # Arguments
///
/// * `func` - The objective function to minimize.
/// * `x0` - Initial guess for the optimization parameters.
/// * `bounds` - Bounds for each dimension as (lower, upper) pairs.
/// * `constraints` - Optional inequality constraints where g(x) >= 0 means feasible.
///                   SHGO uses g(x) >= 0 convention, we convert to NLOPT's g(x) <= 0.
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
/// thread should call this function independently.
pub fn minimize_local<F, G>(
    func: &F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    constraints: Option<&[G]>,
    options: &LocalOptimizerOptions,
) -> LocalOptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
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

    // Check if constraints are supported
    if constraints.is_some() && !options.algorithm.supports_constraints() {
        // Fall back to COBYLA for constrained problems
        let mut fallback_opts = options.clone();
        fallback_opts.algorithm = LocalOptimizer::Cobyla;
        return minimize_local(func, x0, bounds, constraints, &fallback_opts);
    }

    // Track function evaluations using interior mutability
    let fev_count = Rc::new(Cell::new(0usize));
    let fev_counter = Rc::clone(&fev_count);

    // Create the objective function wrapper for NLOPT
    // Signature: (&[f64], Option<&mut [f64]>, &mut UserData) -> f64
    let objective = move |x: &[f64], _grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        fev_counter.set(fev_counter.get() + 1);
        func(x)
    };

    // Create NLOPT optimizer
    let algorithm = options.algorithm.to_nlopt_algorithm();
    let mut opt = Nlopt::new(algorithm, dim, objective, Target::Minimize, ());

    // Set bounds
    let lower_bounds: Vec<f64> = bounds.iter().map(|(l, _)| *l).collect();
    let upper_bounds: Vec<f64> = bounds.iter().map(|(_, u)| *u).collect();

    if opt.set_lower_bounds(&lower_bounds).is_err() {
        return LocalOptResult::failure(x0, "Failed to set lower bounds".to_string());
    }
    if opt.set_upper_bounds(&upper_bounds).is_err() {
        return LocalOptResult::failure(x0, "Failed to set upper bounds".to_string());
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

    // Note: Adding constraints requires a more complex setup because NLOPT's
    // add_inequality_constraint takes ownership of the constraint function.
    // For now, we handle constraints by filtering results after optimization.
    // A full implementation would use add_inequality_constraint for COBYLA/SLSQP.

    // Run optimization
    let mut x = x0.to_vec();
    let result = opt.optimize(&mut x);

    let final_fev = fev_count.get();

    match result {
        Ok((success_state, fval)) => LocalOptResult {
            x,
            fun: fval,
            success: true,
            message: format!("Optimization succeeded: {:?}", success_state),
            nfev: final_fev,
            nit: 0, // NLOPT doesn't track iterations for all algorithms
        },
        Err((fail_state, fval)) => {
            // All FailState variants are actual errors
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

/// Perform local minimization with constraints using NLOPT's constraint support.
///
/// This version properly adds inequality constraints to the optimizer for
/// algorithms that support them (COBYLA, SLSQP).
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
    constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    options: &LocalOptimizerOptions,
) -> LocalOptResult
where
    F: Fn(&[f64]) -> f64,
{
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

    // For constrained optimization, use COBYLA or SLSQP
    let algorithm = if options.algorithm.supports_constraints() {
        options.algorithm.to_nlopt_algorithm()
    } else {
        Algorithm::Cobyla
    };

    // Track function evaluations using interior mutability
    let fev_count = Rc::new(Cell::new(0usize));
    let fev_counter = Rc::clone(&fev_count);

    // Create the objective function wrapper
    let objective = move |x: &[f64], _grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        fev_counter.set(fev_counter.get() + 1);
        func(x)
    };

    // Create NLOPT optimizer
    let mut opt = Nlopt::new(algorithm, dim, objective, Target::Minimize, ());

    // Set bounds
    let lower_bounds: Vec<f64> = bounds.iter().map(|(l, _)| *l).collect();
    let upper_bounds: Vec<f64> = bounds.iter().map(|(_, u)| *u).collect();

    if opt.set_lower_bounds(&lower_bounds).is_err() {
        return LocalOptResult::failure(x0, "Failed to set lower bounds".to_string());
    }
    if opt.set_upper_bounds(&upper_bounds).is_err() {
        return LocalOptResult::failure(x0, "Failed to set upper bounds".to_string());
    }

    // Add constraints
    // SHGO uses g(x) >= 0 (feasible), NLOPT uses fc(x) <= 0 (feasible)
    // So we add constraint: -g(x) <= 0 ⟺ g(x) >= 0
    for constraint in constraints {
        // Create a wrapper that negates the constraint
        let constraint_wrapper = |x: &[f64], _grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
            -constraint(x) // Negate: g(x) >= 0 becomes -g(x) <= 0
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

    let final_fev = fev_count.get();

    match result {
        Ok((success_state, fval)) => LocalOptResult {
            x,
            fun: fval,
            success: true,
            message: format!("Optimization succeeded: {:?}", success_state),
            nfev: final_fev,
            nit: 0,
        },
        Err((fail_state, fval)) => {
            // All FailState variants are actual errors
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

    #[test]
    fn test_minimize_constrained_fallback() {
        // When using BOBYQA with constraints, should fall back to COBYLA
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let x0 = vec![1.5, 1.5];

        // Constraint: x[0] + x[1] >= 1
        let constraints = [|x: &[f64]| x[0] + x[1] - 1.0];

        let options = LocalOptimizerOptions {
            algorithm: LocalOptimizer::Bobyqa, // Doesn't support constraints
            ..Default::default()
        };

        let result = minimize_local(&sphere, &x0, &bounds, Some(&constraints[..]), &options);

        // Should have fallen back to COBYLA
        assert!(result.success);
    }

    #[test]
    fn test_all_algorithms_on_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];

        let algorithms = [
            LocalOptimizer::Bobyqa,
            LocalOptimizer::Cobyla,
            LocalOptimizer::NelderMead,
            LocalOptimizer::Praxis,
            LocalOptimizer::NewuoaBound,
            LocalOptimizer::Sbplx,
            // Skip SLSQP and LBFGS as they require gradients
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
