/**
 * SHGO C++ Example
 * 
 * This example demonstrates how to use the SHGO library from C++.
 * 
 * Compile with:
 *   g++ -std=c++17 -O2 -I../include example.cpp ../target/release/libshgo.a -lpthread -o example_cpp
 * 
 * On Linux you may also need: -ldl -lm
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "shgo.hpp"

// Helper to print results
void print_result(const std::string& name, const shgo::Result& result) {
    std::cout << "\n" << name << " Results:\n";
    std::cout << "  Optimal x = [";
    for (size_t i = 0; i < result.x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << result.x[i];
        if (i < result.x.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "  f(x) = " << std::scientific << std::setprecision(6) << result.fun << "\n";
    std::cout << "  Function evaluations: " << result.nfev << "\n";
    std::cout << "  Local minimizations: " << result.nlfev << "\n";
    std::cout << "  Iterations: " << result.nit << "\n";
    std::cout << "  Success: " << (result.success ? "true" : "false") << "\n";
    std::cout << "  Local minima found: " << result.xl.size() << "\n";
}

int main() {
    std::cout << "=== SHGO C++ Examples ===" << std::endl;
    std::cout << "Library version: " << shgo::version() << std::endl;

    try {
        // Example 1: Simple 2D Sphere function using convenience function
        {
            std::cout << "\n--- Example 1: 2D Sphere Function ---" << std::endl;
            
            auto sphere = [](const std::vector<double>& x) {
                double sum = 0.0;
                for (double xi : x) {
                    sum += xi * xi;
                }
                return sum;
            };

            std::vector<double> lower = {-5.0, -5.0};
            std::vector<double> upper = {5.0, 5.0};
            
            shgo::Options opts;
            opts.maxiter = 3;
            
            auto result = shgo::minimize(sphere, lower, upper, opts);
            print_result("Sphere", result);
        }

        // Example 2: Rosenbrock function using Optimizer class
        {
            std::cout << "\n--- Example 2: 2D Rosenbrock Function ---" << std::endl;
            
            auto rosenbrock = [](const std::vector<double>& x) {
                double sum = 0.0;
                for (size_t i = 0; i < x.size() - 1; ++i) {
                    double t1 = x[i+1] - x[i] * x[i];
                    double t2 = 1.0 - x[i];
                    sum += 100.0 * t1 * t1 + t2 * t2;
                }
                return sum;
            };

            std::vector<double> lower = {-5.0, -5.0};
            std::vector<double> upper = {5.0, 5.0};
            
            shgo::Options opts;
            opts.maxiter = 5;
            
            auto result = shgo::Optimizer(rosenbrock, lower, upper)
                .with_options(opts)
                .minimize();
            
            print_result("Rosenbrock", result);
            std::cout << "  Expected minimum at (1, 1)\n";
        }

        // Example 3: Rastrigin with Sobol sampling
        {
            std::cout << "\n--- Example 3: 2D Rastrigin (Sobol sampling) ---" << std::endl;
            
            auto rastrigin = [](const std::vector<double>& x) {
                const double A = 10.0;
                double sum = A * x.size();
                for (double xi : x) {
                    sum += xi * xi - A * std::cos(2.0 * M_PI * xi);
                }
                return sum;
            };

            std::vector<double> lower = {-5.12, -5.12};
            std::vector<double> upper = {5.12, 5.12};
            
            shgo::Options opts;
            opts.n = 256;
            opts.maxiter = 5;
            opts.sampling_method = shgo::SamplingMethod::Sobol;
            
            auto result = shgo::minimize(rastrigin, lower, upper, opts);
            print_result("Rastrigin", result);
            
            if (!result.xl.empty()) {
                std::cout << "  First local minimum: [" 
                          << result.xl[0].x[0] << ", " << result.xl[0].x[1] 
                          << "] -> " << result.xl[0].fun << "\n";
            }
        }

        // Example 4: 3D optimization with parallel workers
        {
            std::cout << "\n--- Example 4: 3D Sphere (Parallel) ---" << std::endl;
            
            auto sphere3d = [](const std::vector<double>& x) {
                double sum = 0.0;
                for (double xi : x) sum += xi * xi;
                return sum;
            };

            std::vector<double> lower = {-5.0, -5.0, -5.0};
            std::vector<double> upper = {5.0, 5.0, 5.0};
            
            shgo::Options opts;
            opts.maxiter = 3;
            opts.workers = 4;  // Use 4 threads
            
            auto result = shgo::minimize(sphere3d, lower, upper, opts);
            print_result("3D Sphere", result);
        }

        // Example 5: Ackley function
        {
            std::cout << "\n--- Example 5: 2D Ackley Function ---" << std::endl;
            
            auto ackley = [](const std::vector<double>& x) {
                const double a = 20.0;
                const double b = 0.2;
                const double c = 2.0 * M_PI;
                const size_t d = x.size();
                
                double sum1 = 0.0, sum2 = 0.0;
                for (double xi : x) {
                    sum1 += xi * xi;
                    sum2 += std::cos(c * xi);
                }
                
                return -a * std::exp(-b * std::sqrt(sum1 / d)) 
                       - std::exp(sum2 / d) + a + std::exp(1.0);
            };

            std::vector<double> lower = {-5.0, -5.0};
            std::vector<double> upper = {5.0, 5.0};
            
            shgo::Options opts;
            opts.n = 128;
            opts.maxiter = 5;
            
            auto result = shgo::minimize(ackley, lower, upper, opts);
            print_result("Ackley", result);
            std::cout << "  Expected minimum at (0, 0) with f(x) = 0\n";
        }

        std::cout << "\n=== All examples completed successfully ===" << std::endl;

    } catch (const shgo::ShgoException& e) {
        std::cerr << "SHGO Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
