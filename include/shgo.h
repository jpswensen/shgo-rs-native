/*
 * SHGO - Simplicial Homology Global Optimization
 * 
 * A high-performance global optimization library written in Rust
 * with C/C++ bindings.
 * 
 * Version: 0.1.0
 * 
 * Copyright (c) 2024 SHGO Rust Team
 * Licensed under the MIT License
 */

#ifndef SHGO_H
#define SHGO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes returned by SHGO functions
 */
typedef enum ShgoStatus {
    /** Operation completed successfully */
    SHGO_SUCCESS = 0,
    /** Invalid argument provided */
    SHGO_INVALID_ARGUMENT = -1,
    /** Null pointer provided */
    SHGO_NULL_POINTER = -2,
    /** Memory allocation failed */
    SHGO_ALLOCATION_FAILED = -3,
    /** Optimization failed */
    SHGO_OPTIMIZATION_FAILED = -4,
    /** Dimension mismatch */
    SHGO_DIMENSION_MISMATCH = -5,
    /** Invalid bounds (lower >= upper) */
    SHGO_INVALID_BOUNDS = -6,
    /** Unknown error */
    SHGO_UNKNOWN = -99
} ShgoStatus;

/**
 * @brief Sampling methods available for SHGO
 */
typedef enum ShgoSamplingMethod {
    /** Simplicial complex-based sampling (default) */
    SHGO_SAMPLING_SIMPLICIAL = 0,
    /** Sobol sequence quasi-random sampling */
    SHGO_SAMPLING_SOBOL = 1
} ShgoSamplingMethod;

/**
 * @brief Local optimizer algorithms
 */
typedef enum ShgoLocalOptimizer {
    /** BOBYQA - Bound Optimization BY Quadratic Approximation (default) */
    SHGO_LOCAL_BOBYQA = 0,
    /** COBYLA - Constrained Optimization BY Linear Approximations */
    SHGO_LOCAL_COBYLA = 1,
    /** SLSQP - Sequential Least Squares Programming */
    SHGO_LOCAL_SLSQP = 2,
    /** L-BFGS - Limited-memory BFGS */
    SHGO_LOCAL_LBFGS = 3,
    /** Nelder-Mead simplex method */
    SHGO_LOCAL_NELDER_MEAD = 4,
    /** Praxis - Principal axis method */
    SHGO_LOCAL_PRAXIS = 5,
    /** NEWUOA with bounds */
    SHGO_LOCAL_NEWUOA_BOUND = 6,
    /** Subplex - Subspace-searching simplex */
    SHGO_LOCAL_SBPLX = 7
} ShgoLocalOptimizer;

/**
 * @brief Options for SHGO optimization
 * 
 * Use shgo_options_default() to get a default-initialized options struct.
 * 
 * Note: This struct must match ShgoOptions_C in Rust exactly.
 * Fields are ordered to match Rust's repr(C) layout.
 */
typedef struct ShgoOptions {
    /** Number of sampling points (default: 128) */
    size_t n;
    /** Maximum number of iterations (0 = unlimited) */
    size_t maxiter;
    /** Maximum number of function evaluations (0 = unlimited) */
    size_t maxfev;
    /** Maximum time in seconds (0.0 = unlimited) */
    double maxtime;
    /** Function tolerance for local minimization (default: 1e-12) */
    double f_tol;
    /** Display level (0 = silent, 1 = summary, 2 = detailed) */
    size_t disp;
    /** Sampling method */
    ShgoSamplingMethod sampling_method;
    /** Local optimizer algorithm */
    ShgoLocalOptimizer local_optimizer;
    /** Number of worker threads (0 = use all available) */
    size_t workers;
    /** Whether to perform local minimization every iteration */
    bool minimize_every_iter;
} ShgoOptions;

/**
 * @brief A local minimum found during optimization
 */
typedef struct ShgoLocalMinimum {
    /** Pointer to the coordinates (array of length `dim`) */
    double* x;
    /** Function value at this point */
    double fun;
    /** Dimension of the point */
    size_t dim;
} ShgoLocalMinimum;

/**
 * @brief Result of SHGO optimization
 * 
 * Free with shgo_free_result() when done.
 */
typedef struct ShgoResult {
    /** Pointer to optimal point (array of length `dim`) */
    double* x;
    /** Optimal function value */
    double fun;
    /** Dimension of the problem */
    size_t dim;
    /** Number of function evaluations */
    size_t nfev;
    /** Number of local function evaluations */
    size_t nlfev;
    /** Number of iterations */
    size_t nit;
    /** Whether optimization was successful */
    bool success;
    /** Array of all local minima found */
    ShgoLocalMinimum* local_minima;
    /** Pointer to array of function values at local minima (for convenience) */
    double* local_minima_fun;
    /** Number of local minima */
    size_t num_local_minima;
} ShgoResult;

/**
 * @brief Objective function type
 * 
 * @param x Pointer to array of coordinates (length = dim)
 * @param dim Number of dimensions
 * @param user_data User-provided context pointer
 * @return The function value at x
 */
typedef double (*ShgoObjectiveFunc)(const double* x, size_t dim, void* user_data);

/**
 * @brief Constraint function type
 * 
 * Constraint functions should return >= 0 when satisfied (scipy convention).
 * 
 * @param x Pointer to array of coordinates (length = dim)
 * @param dim Number of dimensions
 * @param user_data User-provided context pointer
 * @return Constraint value (>= 0 means satisfied)
 */
typedef double (*ShgoConstraintFunc)(const double* x, size_t dim, void* user_data);

/**
 * @brief Get default SHGO options
 * 
 * Returns an options struct initialized with sensible defaults:
 * - n = 128
 * - maxiter = 0 (unlimited)
 * - maxfev = 0 (unlimited)
 * - maxtime = 0.0 (unlimited)
 * - f_tol = 1e-12
 * - disp = 0 (silent)
 * - sampling_method = SHGO_SAMPLING_SIMPLICIAL
 * - local_optimizer = SHGO_LOCAL_BOBYQA
 * - workers = 0 (use all cores)
 * - minimize_every_iter = true
 * 
 * @return ShgoOptions struct initialized with default values
 */
ShgoOptions shgo_options_default(void);

/**
 * @brief Create a new SHGO optimizer handle
 * 
 * @param objective The objective function to minimize
 * @param user_data User data pointer passed to objective function (can be NULL)
 * @param bounds_lower Array of lower bounds (length = dim)
 * @param bounds_upper Array of upper bounds (length = dim)
 * @param dim Number of dimensions
 * @return Opaque handle to the optimizer, or NULL on failure
 * 
 * @note The returned handle must be freed with shgo_free()
 */
void* shgo_create(ShgoObjectiveFunc objective,
                  void* user_data,
                  const double* bounds_lower,
                  const double* bounds_upper,
                  size_t dim);

/**
 * @brief Set options for the SHGO optimizer
 * 
 * @param handle Handle returned by shgo_create()
 * @param options Pointer to options struct
 * @return SHGO_SUCCESS on success, error code otherwise
 */
ShgoStatus shgo_set_options(void* handle, const ShgoOptions* options);

/**
 * @brief Add a constraint to the optimizer
 * 
 * Constraint functions should return >= 0 when satisfied (scipy convention).
 * 
 * @param handle Handle returned by shgo_create()
 * @param constraint Constraint function (returns >= 0 if satisfied)
 * @param user_data User data passed to constraint function
 * @return SHGO_SUCCESS on success, error code otherwise
 */
ShgoStatus shgo_add_constraint(void* handle,
                               ShgoConstraintFunc constraint,
                               void* user_data);

/**
 * @brief Run the optimization
 * 
 * @param handle Handle returned by shgo_create()
 * @param result Pointer to ShgoResult struct to store results
 * @return SHGO_SUCCESS on success, error code otherwise
 * 
 * @note The result must be freed with shgo_free_result()
 */
ShgoStatus shgo_minimize(void* handle, ShgoResult* result);

/**
 * @brief Free a SHGO result
 * 
 * Frees all memory associated with the result, including the optimal
 * point array and local minima.
 * 
 * @param result Pointer to result to free
 */
void shgo_free_result(ShgoResult* result);

/**
 * @brief Free a SHGO handle
 * 
 * @param handle Handle to free (can be NULL)
 */
void shgo_free(void* handle);

/**
 * @brief Get the library version string
 * 
 * @return Null-terminated version string (e.g., "0.1.0")
 */
const char* shgo_version(void);

/**
 * @brief Convenience function: minimize in a single call
 * 
 * This is a simplified API that creates a handle, sets options,
 * runs optimization, and cleans up in one call.
 * 
 * @param objective The objective function to minimize
 * @param user_data User data pointer passed to objective function (can be NULL)
 * @param bounds_lower Array of lower bounds (length = dim)
 * @param bounds_upper Array of upper bounds (length = dim)
 * @param dim Number of dimensions
 * @param options Options (can be NULL for defaults)
 * @param result Pointer to store results
 * @return SHGO_SUCCESS on success, error code otherwise
 * 
 * @note The result must be freed with shgo_free_result()
 */
ShgoStatus shgo_minimize_simple(ShgoObjectiveFunc objective,
                                void* user_data,
                                const double* bounds_lower,
                                const double* bounds_upper,
                                size_t dim,
                                const ShgoOptions* options,
                                ShgoResult* result);

#ifdef __cplusplus
}
#endif

#endif /* SHGO_H */
