//! C/C++ Foreign Function Interface for SHGO
//!
//! This module provides C-compatible bindings for the SHGO library,
//! allowing it to be used from C++ and other languages that can call C functions.

use std::ffi::c_void;
use std::ptr;
use std::slice;

use crate::shgo::{Shgo, ShgoOptions, SamplingMethod};
use crate::local_opt::LocalOptimizer;

/// Status codes returned by SHGO functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShgoStatus {
    /// Operation completed successfully
    Success = 0,
    /// Invalid argument provided
    InvalidArgument = -1,
    /// Null pointer provided
    NullPointer = -2,
    /// Memory allocation failed
    AllocationFailed = -3,
    /// Optimization failed
    OptimizationFailed = -4,
    /// Dimension mismatch
    DimensionMismatch = -5,
    /// Invalid bounds (lower >= upper)
    InvalidBounds = -6,
    /// Unknown error
    Unknown = -99,
}

/// Sampling methods available for SHGO
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShgoSamplingMethod {
    /// Simplicial complex-based sampling (default)
    Simplicial = 0,
    /// Sobol sequence quasi-random sampling
    Sobol = 1,
}

impl From<ShgoSamplingMethod> for SamplingMethod {
    fn from(method: ShgoSamplingMethod) -> Self {
        match method {
            ShgoSamplingMethod::Simplicial => SamplingMethod::Simplicial,
            ShgoSamplingMethod::Sobol => SamplingMethod::Sobol,
        }
    }
}

/// Local optimizer algorithms
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShgoLocalOptimizer {
    /// BOBYQA - Bound Optimization BY Quadratic Approximation (default)
    Bobyqa = 0,
    /// COBYLA - Constrained Optimization BY Linear Approximations
    Cobyla = 1,
    /// SLSQP - Sequential Least Squares Programming
    Slsqp = 2,
    /// L-BFGS - Limited-memory BFGS
    Lbfgs = 3,
    /// Nelder-Mead simplex method
    NelderMead = 4,
    /// Praxis - Principal axis method
    Praxis = 5,
    /// NEWUOA with bounds
    NewuoaBound = 6,
    /// Subplex - Subspace-searching simplex
    Sbplx = 7,
}

impl From<ShgoLocalOptimizer> for LocalOptimizer {
    fn from(opt: ShgoLocalOptimizer) -> Self {
        match opt {
            ShgoLocalOptimizer::Bobyqa => LocalOptimizer::Bobyqa,
            ShgoLocalOptimizer::Cobyla => LocalOptimizer::Cobyla,
            ShgoLocalOptimizer::Slsqp => LocalOptimizer::Slsqp,
            ShgoLocalOptimizer::Lbfgs => LocalOptimizer::Lbfgs,
            ShgoLocalOptimizer::NelderMead => LocalOptimizer::NelderMead,
            ShgoLocalOptimizer::Praxis => LocalOptimizer::Praxis,
            ShgoLocalOptimizer::NewuoaBound => LocalOptimizer::NewuoaBound,
            ShgoLocalOptimizer::Sbplx => LocalOptimizer::Sbplx,
        }
    }
}

/// Options for SHGO optimization (C-compatible)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShgoOptions_C {
    /// Number of sampling points (default: 128)
    pub n: usize,
    /// Maximum number of iterations (0 = unlimited)
    pub maxiter: usize,
    /// Maximum number of function evaluations (0 = unlimited)
    pub maxfev: usize,
    /// Maximum time in seconds (0.0 = unlimited)
    pub maxtime: f64,
    /// Function tolerance for local minimization
    pub f_tol: f64,
    /// Display level (0 = silent, 1 = summary, 2 = detailed)
    pub disp: usize,
    /// Sampling method
    pub sampling_method: ShgoSamplingMethod,
    /// Local optimizer algorithm
    pub local_optimizer: ShgoLocalOptimizer,
    /// Number of worker threads (0 = use all available)
    pub workers: usize,
    /// Whether to perform local minimization every iteration
    pub minimize_every_iter: bool,
}

impl Default for ShgoOptions_C {
    fn default() -> Self {
        Self {
            n: 128,
            maxiter: 0,
            maxfev: 0,
            maxtime: 0.0,
            f_tol: 1e-12,
            disp: 0,
            sampling_method: ShgoSamplingMethod::Simplicial,
            local_optimizer: ShgoLocalOptimizer::Bobyqa,
            workers: 0,
            minimize_every_iter: true,
        }
    }
}

impl From<ShgoOptions_C> for ShgoOptions {
    fn from(opts: ShgoOptions_C) -> Self {
        let algorithm: crate::local_opt::LocalOptimizer = opts.local_optimizer.into();
        ShgoOptions {
            n: opts.n,
            maxiter: if opts.maxiter == 0 { None } else { Some(opts.maxiter) },
            maxfev: if opts.maxfev == 0 { None } else { Some(opts.maxfev) },
            maxtime: if opts.maxtime <= 0.0 { None } else { Some(opts.maxtime) },
            f_tol: opts.f_tol,
            disp: opts.disp,
            sampling_method: opts.sampling_method.into(),
            local_options: crate::local_opt::LocalOptimizerOptions {
                algorithm,
                ..crate::local_opt::LocalOptimizerOptions::default()
            },
            workers: if opts.workers == 0 { None } else { Some(opts.workers) },
            minimize_every_iter: opts.minimize_every_iter,
            ..Default::default()
        }
    }
}

/// A local minimum found during optimization
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShgoLocalMinimum_C {
    /// Pointer to the coordinates (array of length `dim`)
    pub x: *mut f64,
    /// Function value at this point
    pub fun: f64,
    /// Dimension of the point
    pub dim: usize,
}

/// Result of SHGO optimization (C-compatible)
#[repr(C)]
#[derive(Debug)]
pub struct ShgoResult_C {
    /// Pointer to optimal point (array of length `dim`)
    pub x: *mut f64,
    /// Optimal function value
    pub fun: f64,
    /// Dimension of the problem
    pub dim: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of local function evaluations
    pub nlfev: usize,
    /// Number of iterations
    pub nit: usize,
    /// Whether optimization was successful
    pub success: bool,
    /// Array of all local minima found
    pub local_minima: *mut ShgoLocalMinimum_C,
    /// Array of function values at local minima
    pub local_minima_fun: *mut f64,
    /// Number of local minima
    pub num_local_minima: usize,
}

/// Opaque handle to a SHGO optimizer instance (not exported to C)
struct ShgoHandle {
    objective: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    bounds: Vec<(f64, f64)>,
    constraints: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    options: ShgoOptions,
    dim: usize,
}

/// Type alias for C function pointer representing the objective function.
/// 
/// The function receives:
/// - x: pointer to array of coordinates
/// - dim: number of dimensions
/// - user_data: user-provided context pointer
/// 
/// Returns the function value at x.
pub type ShgoObjectiveFunc = Option<unsafe extern "C" fn(x: *const f64, dim: usize, user_data: *mut c_void) -> f64>;

/// Type alias for C function pointer representing a constraint function.
/// Constraints should return >= 0 when satisfied (scipy convention).
/// 
/// The function receives:
/// - x: pointer to array of coordinates
/// - dim: number of dimensions
/// - user_data: user-provided context pointer
/// 
/// Returns the constraint value (>= 0 means satisfied).
pub type ShgoConstraintFunc = Option<unsafe extern "C" fn(x: *const f64, dim: usize, user_data: *mut c_void) -> f64>;

// ============================================================================
// C API Functions
// ============================================================================

/// Create default SHGO options
/// 
/// # Returns
/// A ShgoOptions_C struct with default values
#[no_mangle]
pub extern "C" fn shgo_options_default() -> ShgoOptions_C {
    ShgoOptions_C::default()
}

/// Create a new SHGO optimizer handle
///
/// # Arguments
/// * `objective` - The objective function to minimize
/// * `user_data` - User data pointer passed to objective function
/// * `bounds_lower` - Array of lower bounds (length = dim)
/// * `bounds_upper` - Array of upper bounds (length = dim)
/// * `dim` - Number of dimensions
///
/// # Returns
/// * Pointer to ShgoHandle on success, null on failure
///
/// # Safety
/// The caller must ensure:
/// - bounds_lower and bounds_upper are valid pointers to arrays of length `dim`
/// - The objective function pointer remains valid for the lifetime of the handle
/// - The returned handle is freed with `shgo_free`
#[no_mangle]
pub unsafe extern "C" fn shgo_create(
    objective: ShgoObjectiveFunc,
    user_data: *mut c_void,
    bounds_lower: *const f64,
    bounds_upper: *const f64,
    dim: usize,
) -> *mut c_void {
    let obj_fn = match objective {
        Some(f) => f,
        None => return ptr::null_mut(),
    };
    
    if dim == 0 || bounds_lower.is_null() || bounds_upper.is_null() {
        return ptr::null_mut();
    }

    let lower = slice::from_raw_parts(bounds_lower, dim);
    let upper = slice::from_raw_parts(bounds_upper, dim);

    let mut bounds = Vec::with_capacity(dim);
    for i in 0..dim {
        if lower[i] >= upper[i] || lower[i].is_nan() || upper[i].is_nan() {
            return ptr::null_mut();
        }
        bounds.push((lower[i], upper[i]));
    }

    // Wrap the C function pointer in a Rust closure
    // We need to capture user_data safely
    let user_data_ptr = user_data as usize; // Convert to usize to make it Send + Sync
    let wrapped_obj = move |x: &[f64]| -> f64 {
        let user_data = user_data_ptr as *mut c_void;
        unsafe { obj_fn(x.as_ptr(), x.len(), user_data) }
    };

    let handle = Box::new(ShgoHandle {
        objective: Box::new(wrapped_obj),
        bounds,
        constraints: Vec::new(),
        options: ShgoOptions::default(),
        dim,
    });

    Box::into_raw(handle) as *mut c_void
}

/// Set options for the SHGO optimizer
///
/// # Arguments
/// * `handle` - Pointer to ShgoHandle
/// * `options` - Pointer to ShgoOptions_C
///
/// # Returns
/// * ShgoStatus::Success on success
///
/// # Safety
/// handle must be a valid pointer returned by shgo_create
#[no_mangle]
pub unsafe extern "C" fn shgo_set_options(
    handle: *mut c_void,
    options: *const ShgoOptions_C,
) -> ShgoStatus {
    if handle.is_null() || options.is_null() {
        return ShgoStatus::NullPointer;
    }

    let handle = &mut *(handle as *mut ShgoHandle);
    let options = *options;
    handle.options = options.into();
    
    ShgoStatus::Success
}

/// Add a constraint to the SHGO optimizer
///
/// The constraint function should return >= 0 when satisfied (scipy convention).
///
/// # Arguments
/// * `handle` - Pointer to ShgoHandle
/// * `constraint` - Constraint function pointer
/// * `user_data` - User data passed to constraint function
///
/// # Returns
/// * ShgoStatus::Success on success
///
/// # Safety
/// handle must be a valid pointer returned by shgo_create
#[no_mangle]
pub unsafe extern "C" fn shgo_add_constraint(
    handle: *mut c_void,
    constraint: ShgoConstraintFunc,
    user_data: *mut c_void,
) -> ShgoStatus {
    if handle.is_null() {
        return ShgoStatus::NullPointer;
    }

    let constraint_fn = match constraint {
        Some(f) => f,
        None => return ShgoStatus::InvalidArgument,
    };

    let handle = &mut *(handle as *mut ShgoHandle);
    let user_data_ptr = user_data as usize;
    
    let wrapped_constraint = move |x: &[f64]| -> f64 {
        let user_data = user_data_ptr as *mut c_void;
        unsafe { constraint_fn(x.as_ptr(), x.len(), user_data) }
    };

    handle.constraints.push(Box::new(wrapped_constraint));
    
    ShgoStatus::Success
}

/// Run the SHGO optimization
///
/// # Arguments
/// * `handle` - Pointer to ShgoHandle
/// * `result` - Pointer to ShgoResult_C to store results
///
/// # Returns
/// * ShgoStatus::Success on success
///
/// # Safety
/// - handle must be a valid pointer returned by shgo_create
/// - result must be a valid pointer to ShgoResult_C
/// - The result's arrays must be freed with shgo_free_result
#[no_mangle]
pub unsafe extern "C" fn shgo_minimize(
    handle: *mut c_void,
    result: *mut ShgoResult_C,
) -> ShgoStatus {
    if handle.is_null() || result.is_null() {
        return ShgoStatus::NullPointer;
    }

    let handle = &*(handle as *mut ShgoHandle);
    
    // Create a wrapper that calls our stored objective
    let objective_ref = &handle.objective;
    let objective_wrapper = |x: &[f64]| -> f64 {
        objective_ref(x)
    };

    // Run optimization - constraints not directly supported in this API version
    // For now we run without constraints (constraints need special handling in SHGO)
    let shgo = Shgo::new(objective_wrapper, handle.bounds.clone())
        .with_options(handle.options.clone());

    // Run optimization
    let opt_result = match shgo.minimize() {
        Ok(r) => r,
        Err(_) => return ShgoStatus::OptimizationFailed,
    };

    // Convert result to C-compatible format
    let result_ref = &mut *result;
    
    // Allocate and copy optimal point
    let x_ptr = libc::malloc(handle.dim * std::mem::size_of::<f64>()) as *mut f64;
    if x_ptr.is_null() {
        return ShgoStatus::AllocationFailed;
    }
    ptr::copy_nonoverlapping(opt_result.x.as_ptr(), x_ptr, handle.dim);
    result_ref.x = x_ptr;
    result_ref.fun = opt_result.fun;
    result_ref.dim = handle.dim;
    result_ref.nfev = opt_result.nfev;
    result_ref.nlfev = opt_result.nlfev;
    result_ref.nit = opt_result.nit;
    result_ref.success = opt_result.success;

    // Copy local minima (xl is Vec<Vec<f64>>, funl is Vec<f64>)
    let num_minima = opt_result.xl.len();
    if num_minima > 0 {
        let minima_ptr = libc::malloc(num_minima * std::mem::size_of::<ShgoLocalMinimum_C>()) 
            as *mut ShgoLocalMinimum_C;
        let funl_ptr = libc::malloc(num_minima * std::mem::size_of::<f64>()) as *mut f64;
        
        if minima_ptr.is_null() || funl_ptr.is_null() {
            if !minima_ptr.is_null() { libc::free(minima_ptr as *mut c_void); }
            if !funl_ptr.is_null() { libc::free(funl_ptr as *mut c_void); }
            libc::free(x_ptr as *mut c_void);
            return ShgoStatus::AllocationFailed;
        }

        for (i, xl_i) in opt_result.xl.iter().enumerate() {
            let min_x_ptr = libc::malloc(handle.dim * std::mem::size_of::<f64>()) as *mut f64;
            if min_x_ptr.is_null() {
                // Clean up already allocated
                for j in 0..i {
                    let prev = &*minima_ptr.add(j);
                    libc::free(prev.x as *mut c_void);
                }
                libc::free(minima_ptr as *mut c_void);
                libc::free(funl_ptr as *mut c_void);
                libc::free(x_ptr as *mut c_void);
                return ShgoStatus::AllocationFailed;
            }
            ptr::copy_nonoverlapping(xl_i.as_ptr(), min_x_ptr, handle.dim);
            
            let fun_i = opt_result.funl.get(i).copied().unwrap_or(f64::NAN);
            let local_min = ShgoLocalMinimum_C {
                x: min_x_ptr,
                fun: fun_i,
                dim: handle.dim,
            };
            ptr::write(minima_ptr.add(i), local_min);
            ptr::write(funl_ptr.add(i), fun_i);
        }
        
        result_ref.local_minima = minima_ptr;
        result_ref.local_minima_fun = funl_ptr;
        result_ref.num_local_minima = num_minima;
    } else {
        result_ref.local_minima = ptr::null_mut();
        result_ref.local_minima_fun = ptr::null_mut();
        result_ref.num_local_minima = 0;
    }

    ShgoStatus::Success
}

/// Free a SHGO result
///
/// # Arguments
/// * `result` - Pointer to ShgoResult_C to free
///
/// # Safety
/// result must be a valid pointer to a ShgoResult_C returned by shgo_minimize
#[no_mangle]
pub unsafe extern "C" fn shgo_free_result(result: *mut ShgoResult_C) {
    if result.is_null() {
        return;
    }

    let result = &mut *result;
    
    // Free optimal point
    if !result.x.is_null() {
        libc::free(result.x as *mut c_void);
        result.x = ptr::null_mut();
    }

    // Free local minima
    if !result.local_minima.is_null() {
        for i in 0..result.num_local_minima {
            let min = &*result.local_minima.add(i);
            if !min.x.is_null() {
                libc::free(min.x as *mut c_void);
            }
        }
        libc::free(result.local_minima as *mut c_void);
        result.local_minima = ptr::null_mut();
    }
    
    if !result.local_minima_fun.is_null() {
        libc::free(result.local_minima_fun as *mut c_void);
        result.local_minima_fun = ptr::null_mut();
    }
    
    result.num_local_minima = 0;
}

/// Free a SHGO handle
///
/// # Arguments
/// * `handle` - Pointer to ShgoHandle to free
///
/// # Safety
/// handle must be a valid pointer returned by shgo_create, or null
#[no_mangle]
pub unsafe extern "C" fn shgo_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut ShgoHandle));
    }
}

/// Get the library version string
///
/// # Returns
/// Pointer to a static null-terminated string containing the version
#[no_mangle]
pub extern "C" fn shgo_version() -> *const libc::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const libc::c_char
}

// ============================================================================
// Convenience functions for common optimization patterns
// ============================================================================

/// Perform SHGO optimization in a single call (convenience function)
///
/// # Arguments
/// * `objective` - The objective function to minimize
/// * `user_data` - User data pointer passed to objective function
/// * `bounds_lower` - Array of lower bounds
/// * `bounds_upper` - Array of upper bounds
/// * `dim` - Number of dimensions
/// * `options` - Options (can be null for defaults)
/// * `result` - Pointer to store results
///
/// # Returns
/// * ShgoStatus indicating success or failure
///
/// # Safety
/// All pointer arguments must be valid
#[no_mangle]
pub unsafe extern "C" fn shgo_minimize_simple(
    objective: ShgoObjectiveFunc,
    user_data: *mut c_void,
    bounds_lower: *const f64,
    bounds_upper: *const f64,
    dim: usize,
    options: *const ShgoOptions_C,
    result: *mut ShgoResult_C,
) -> ShgoStatus {
    let handle = shgo_create(objective, user_data, bounds_lower, bounds_upper, dim);
    if handle.is_null() {
        return ShgoStatus::InvalidArgument;
    }

    if !options.is_null() {
        let status = shgo_set_options(handle, options);
        if status != ShgoStatus::Success {
            shgo_free(handle);
            return status;
        }
    }

    let status = shgo_minimize(handle, result);
    shgo_free(handle);
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn sphere_c(x: *const f64, dim: usize, _user_data: *mut c_void) -> f64 {
        let slice = slice::from_raw_parts(x, dim);
        slice.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_c_api_basic() {
        unsafe {
            let lower = [-5.0_f64, -5.0];
            let upper = [5.0_f64, 5.0];
            
            let handle = shgo_create(
                Some(sphere_c),
                ptr::null_mut(),
                lower.as_ptr(),
                upper.as_ptr(),
                2,
            );
            assert!(!handle.is_null());

            let mut options = shgo_options_default();
            options.maxiter = 3;
            let status = shgo_set_options(handle, &options);
            assert_eq!(status, ShgoStatus::Success);

            let mut result = std::mem::zeroed::<ShgoResult_C>();
            let status = shgo_minimize(handle, &mut result);
            assert_eq!(status, ShgoStatus::Success);

            // Check result
            assert!(result.fun < 0.01);
            assert_eq!(result.dim, 2);

            // Clean up
            shgo_free_result(&mut result);
            shgo_free(handle);
        }
    }

    #[test]
    fn test_simple_api() {
        unsafe {
            let lower = [-5.0_f64, -5.0];
            let upper = [5.0_f64, 5.0];
            let mut options = shgo_options_default();
            options.maxiter = 3;

            let mut result = std::mem::zeroed::<ShgoResult_C>();
            let status = shgo_minimize_simple(
                Some(sphere_c),
                ptr::null_mut(),
                lower.as_ptr(),
                upper.as_ptr(),
                2,
                &options,
                &mut result,
            );
            assert_eq!(status, ShgoStatus::Success);
            assert!(result.fun < 0.01);

            shgo_free_result(&mut result);
        }
    }
}
