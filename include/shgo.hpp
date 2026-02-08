/*
 * SHGO - Simplicial Homology Global Optimization
 * 
 * C++ wrapper for the SHGO library
 * 
 * Version: 0.1.0
 * 
 * Copyright (c) 2024 SHGO Rust Team
 * Licensed under the MIT License
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

// Include the C header for the raw API
#include "shgo.h"

namespace shgo {

/**
 * @brief Status codes returned by SHGO functions
 */
enum class Status {
    Success = SHGO_SUCCESS,
    InvalidArgument = SHGO_INVALID_ARGUMENT,
    NullPointer = SHGO_NULL_POINTER,
    AllocationFailed = SHGO_ALLOCATION_FAILED,
    OptimizationFailed = SHGO_OPTIMIZATION_FAILED,
    DimensionMismatch = SHGO_DIMENSION_MISMATCH,
    InvalidBounds = SHGO_INVALID_BOUNDS,
    Unknown = SHGO_UNKNOWN
};

/**
 * @brief Sampling methods available for SHGO
 */
enum class SamplingMethod {
    Simplicial = SHGO_SAMPLING_SIMPLICIAL,
    Sobol = SHGO_SAMPLING_SOBOL
};

/**
 * @brief Local optimizer algorithms
 */
enum class LocalOptimizer {
    Bobyqa = SHGO_LOCAL_BOBYQA,
    Cobyla = SHGO_LOCAL_COBYLA,
    Slsqp = SHGO_LOCAL_SLSQP,
    Lbfgs = SHGO_LOCAL_LBFGS,
    NelderMead = SHGO_LOCAL_NELDER_MEAD,
    Praxis = SHGO_LOCAL_PRAXIS,
    NewuoaBound = SHGO_LOCAL_NEWUOA_BOUND,
    Sbplx = SHGO_LOCAL_SBPLX
};

/**
 * @brief Exception class for SHGO errors
 */
class ShgoException : public std::runtime_error {
public:
    explicit ShgoException(Status status, const char* message = "SHGO error")
        : std::runtime_error(message), status_(status) {}
    
    Status status() const noexcept { return status_; }

private:
    Status status_;
};

/**
 * @brief A local minimum found during optimization
 */
struct LocalMinimum {
    std::vector<double> x;  ///< Coordinates of the minimum
    double fun;             ///< Function value at this point
    
    LocalMinimum(const double* x_ptr, size_t dim, double f)
        : x(x_ptr, x_ptr + dim), fun(f) {}
};

/**
 * @brief Result of SHGO optimization
 */
struct Result {
    std::vector<double> x;              ///< Optimal point
    double fun;                          ///< Optimal function value
    size_t nfev;                         ///< Number of function evaluations
    size_t nlfev;                        ///< Number of local function evaluations
    size_t nit;                          ///< Number of iterations
    bool success;                        ///< Whether optimization was successful
    std::vector<LocalMinimum> xl;        ///< All local minima found
    
    Result() : fun(0), nfev(0), nlfev(0), nit(0), success(false) {}
};

/**
 * @brief Options for SHGO optimization
 */
struct Options {
    size_t n = 128;                                      ///< Number of sampling points
    size_t maxiter = 0;                                  ///< Maximum iterations (0 = unlimited)
    size_t maxfev = 0;                                   ///< Maximum function evaluations (0 = unlimited)
    double maxtime = 0.0;                                ///< Maximum time in seconds (0 = unlimited)
    double f_tol = 1e-12;                                ///< Function tolerance
    size_t disp = 0;                                     ///< Display level (0=silent, 1=summary, 2=detailed)
    SamplingMethod sampling_method = SamplingMethod::Simplicial;
    LocalOptimizer local_optimizer = LocalOptimizer::Bobyqa;
    size_t workers = 0;                                  ///< Number of worker threads (0 = all available)
    bool minimize_every_iter = true;                     ///< Local minimize every iteration
    
    /// Convert to C struct
    ShgoOptions to_c() const {
        ShgoOptions opts;
        opts.n = n;
        opts.maxiter = maxiter;
        opts.maxfev = maxfev;
        opts.maxtime = maxtime;
        opts.f_tol = f_tol;
        opts.disp = disp;
        opts.sampling_method = static_cast<ShgoSamplingMethod>(sampling_method);
        opts.local_optimizer = static_cast<ShgoLocalOptimizer>(local_optimizer);
        opts.workers = workers;
        opts.minimize_every_iter = minimize_every_iter;
        return opts;
    }
};

// Thread-local storage for callback handling
namespace detail {
    inline thread_local std::vector<double> temp_x_;
}

/**
 * @brief SHGO Optimizer class
 * 
 * High-level C++ wrapper for the SHGO global optimization algorithm.
 * 
 * Example usage:
 * @code
 * // Define objective function
 * auto sphere = [](const std::vector<double>& x) {
 *     double sum = 0;
 *     for (double xi : x) sum += xi * xi;
 *     return sum;
 * };
 * 
 * // Set up bounds
 * std::vector<double> lower = {-5.0, -5.0};
 * std::vector<double> upper = {5.0, 5.0};
 * 
 * // Create optimizer and minimize
 * shgo::Optimizer opt(sphere, lower, upper);
 * shgo::Result result = opt.minimize();
 * 
 * std::cout << "Minimum: " << result.fun << std::endl;
 * @endcode
 */
class Optimizer {
public:
    using ObjectiveFunc = std::function<double(const std::vector<double>&)>;

    /**
     * @brief Construct a new Optimizer
     * 
     * @param objective The objective function to minimize
     * @param lower_bounds Lower bounds for each dimension
     * @param upper_bounds Upper bounds for each dimension
     */
    Optimizer(ObjectiveFunc objective,
              const std::vector<double>& lower_bounds,
              const std::vector<double>& upper_bounds)
        : objective_(std::move(objective))
        , lower_bounds_(lower_bounds)
        , upper_bounds_(upper_bounds)
        , dim_(lower_bounds.size())
    {
        if (lower_bounds_.size() != upper_bounds_.size()) {
            throw ShgoException(Status::DimensionMismatch, 
                "Lower and upper bounds must have same dimension");
        }
        if (dim_ == 0) {
            throw ShgoException(Status::InvalidArgument,
                "Dimension must be at least 1");
        }
    }

    /**
     * @brief Set optimization options
     */
    Optimizer& with_options(const Options& opts) {
        options_ = opts;
        return *this;
    }

    /**
     * @brief Run the optimization
     * 
     * @return Result containing the optimal point and function value
     */
    Result minimize() {
        // Create the C handle
        void* handle = shgo_create(
            c_objective_wrapper,
            this,
            lower_bounds_.data(),
            upper_bounds_.data(),
            dim_
        );
        
        if (!handle) {
            throw ShgoException(Status::InvalidArgument, 
                "Failed to create SHGO handle");
        }

        // Set options
        ShgoOptions c_opts = options_.to_c();
        ShgoStatus status = shgo_set_options(handle, &c_opts);
        if (status != SHGO_SUCCESS) {
            shgo_free(handle);
            throw ShgoException(static_cast<Status>(status), 
                "Failed to set options");
        }

        // Run optimization
        ShgoResult c_result = {};
        status = shgo_minimize(handle, &c_result);
        shgo_free(handle);

        if (status != SHGO_SUCCESS) {
            throw ShgoException(static_cast<Status>(status),
                "Optimization failed");
        }

        // Convert result
        Result result;
        if (c_result.x && c_result.dim > 0) {
            result.x = std::vector<double>(c_result.x, c_result.x + c_result.dim);
        }
        result.fun = c_result.fun;
        result.nfev = c_result.nfev;
        result.nlfev = c_result.nlfev;
        result.nit = c_result.nit;
        result.success = c_result.success;

        // Copy local minima
        for (size_t i = 0; i < c_result.num_local_minima; ++i) {
            const ShgoLocalMinimum& lm = c_result.local_minima[i];
            result.xl.emplace_back(lm.x, lm.dim, lm.fun);
        }

        shgo_free_result(&c_result);
        return result;
    }

private:
    ObjectiveFunc objective_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    Options options_;
    size_t dim_;

    static double c_objective_wrapper(const double* x, size_t dim, void* user_data) {
        Optimizer* self = static_cast<Optimizer*>(user_data);
        detail::temp_x_.assign(x, x + dim);
        return self->objective_(detail::temp_x_);
    }
};

/**
 * @brief Convenience function to minimize in a single call
 * 
 * @param objective The objective function
 * @param lower_bounds Lower bounds
 * @param upper_bounds Upper bounds
 * @param options Optional optimization options
 * @return Result containing the optimal point and function value
 */
inline Result minimize(
    std::function<double(const std::vector<double>&)> objective,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    const Options& options = Options())
{
    return Optimizer(std::move(objective), lower_bounds, upper_bounds)
        .with_options(options)
        .minimize();
}

/**
 * @brief Get the library version
 */
inline const char* version() {
    return shgo_version();
}

} // namespace shgo
