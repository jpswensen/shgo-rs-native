/**
 * SHGO C Example
 * 
 * This example demonstrates how to use the SHGO library from C.
 * 
 * Compile with:
 *   gcc -O2 -I../include -L../target/release -lshgo -o example_c example.c
 * 
 * On macOS, you may need to set DYLD_LIBRARY_PATH:
 *   export DYLD_LIBRARY_PATH=../target/release:$DYLD_LIBRARY_PATH
 * 
 * Or use the static library:
 *   gcc -O2 -I../include example.c ../target/release/libshgo.a -lpthread -ldl -lm -o example_c
 */

#include <stdio.h>
#include <math.h>
#include "shgo.h"

/**
 * Sphere function: f(x) = sum(x_i^2)
 * Global minimum at (0, 0, ..., 0) with f(x) = 0
 */
double sphere(const double* x, size_t dim, void* user_data) {
    (void)user_data;  /* unused */
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

/**
 * Rosenbrock function
 * Global minimum at (1, 1, ..., 1) with f(x) = 0
 */
double rosenbrock(const double* x, size_t dim, void* user_data) {
    (void)user_data;
    double sum = 0.0;
    for (size_t i = 0; i < dim - 1; i++) {
        double t1 = x[i+1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    return sum;
}

/**
 * Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
 * Global minimum at (0, 0, ..., 0) with f(x) = 0
 */
double rastrigin(const double* x, size_t dim, void* user_data) {
    (void)user_data;
    const double A = 10.0;
    double sum = A * dim;
    for (size_t i = 0; i < dim; i++) {
        sum += x[i] * x[i] - A * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

void print_result(const char* name, const ShgoResult* result) {
    printf("\n%s Results:\n", name);
    printf("  Optimal x = [");
    for (size_t i = 0; i < result->dim; i++) {
        printf("%.6f", result->x[i]);
        if (i < result->dim - 1) printf(", ");
    }
    printf("]\n");
    printf("  f(x) = %.6e\n", result->fun);
    printf("  Function evaluations: %zu\n", result->nfev);
    printf("  Local minimizations: %zu\n", result->nlfev);
    printf("  Iterations: %zu\n", result->nit);
    printf("  Success: %s\n", result->success ? "true" : "false");
    printf("  Local minima found: %zu\n", result->num_local_minima);
}

int main() {
    printf("=== SHGO C Examples ===\n");
    printf("Library version: %s\n", shgo_version());

    /* Example 1: Using the simple API */
    {
        printf("\n--- Example 1: 2D Sphere (Simple API) ---\n");
        
        double lower[] = {-5.0, -5.0};
        double upper[] = {5.0, 5.0};
        
        ShgoOptions options = shgo_options_default();
        options.maxiter = 3;
        
        ShgoResult result = {0};
        ShgoStatus status = shgo_minimize_simple(
            sphere,
            NULL,  /* user_data */
            lower,
            upper,
            2,     /* dim */
            &options,
            &result
        );
        
        if (status == SHGO_SUCCESS) {
            print_result("Sphere", &result);
            shgo_free_result(&result);
        } else {
            printf("Optimization failed with status: %d\n", status);
        }
    }

    /* Example 2: Using the full API for more control */
    {
        printf("\n--- Example 2: 2D Rosenbrock (Full API) ---\n");
        
        double lower[] = {-5.0, -5.0};
        double upper[] = {5.0, 5.0};
        
        /* Create handle */
        void* handle = shgo_create(rosenbrock, NULL, lower, upper, 2);
        if (!handle) {
            printf("Failed to create optimizer\n");
            return 1;
        }
        
        /* Set options */
        ShgoOptions options = shgo_options_default();
        options.maxiter = 5;
        shgo_set_options(handle, &options);
        
        /* Run optimization */
        ShgoResult result = {0};
        ShgoStatus status = shgo_minimize(handle, &result);
        
        if (status == SHGO_SUCCESS) {
            print_result("Rosenbrock", &result);
            shgo_free_result(&result);
        } else {
            printf("Optimization failed with status: %d\n", status);
        }
        
        /* Free handle */
        shgo_free(handle);
    }

    /* Example 3: Rastrigin with Sobol sampling */
    {
        printf("\n--- Example 3: 2D Rastrigin (Sobol sampling) ---\n");
        
        double lower[] = {-5.12, -5.12};
        double upper[] = {5.12, 5.12};
        
        ShgoOptions options = shgo_options_default();
        options.n = 256;
        options.maxiter = 5;
        options.sampling_method = SHGO_SAMPLING_SOBOL;
        
        ShgoResult result = {0};
        ShgoStatus status = shgo_minimize_simple(
            rastrigin, NULL, lower, upper, 2, &options, &result
        );
        
        if (status == SHGO_SUCCESS) {
            print_result("Rastrigin", &result);
            
            /* Print some local minima */
            if (result.num_local_minima > 0) {
                printf("  First local minimum: [%.6f, %.6f] -> %.6f\n",
                       result.local_minima[0].x[0],
                       result.local_minima[0].x[1],
                       result.local_minima[0].fun);
            }
            
            shgo_free_result(&result);
        } else {
            printf("Optimization failed with status: %d\n", status);
        }
    }

    /* Example 4: 3D optimization with parallel workers */
    {
        printf("\n--- Example 4: 3D Sphere (Parallel) ---\n");
        
        double lower[] = {-5.0, -5.0, -5.0};
        double upper[] = {5.0, 5.0, 5.0};
        
        ShgoOptions options = shgo_options_default();
        options.maxiter = 3;
        options.workers = 4;  /* Use 4 threads */
        
        ShgoResult result = {0};
        ShgoStatus status = shgo_minimize_simple(
            sphere, NULL, lower, upper, 3, &options, &result
        );
        
        if (status == SHGO_SUCCESS) {
            print_result("3D Sphere", &result);
            shgo_free_result(&result);
        } else {
            printf("Optimization failed with status: %d\n", status);
        }
    }

    printf("\n=== All examples completed ===\n");
    return 0;
}
