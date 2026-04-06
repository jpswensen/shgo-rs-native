//! # SHGO-RS: Simplicial Homology Global Optimization in Rust
//!
//! A high-performance Rust implementation of the SHGO algorithm for finding
//! global minima of functions.
//!
//! ## Overview
//!
//! SHGO (Simplicial Homology Global Optimization) is a global optimization
//! algorithm that uses concepts from algebraic topology (simplicial homology)
//! to find all local minima of a function, with theoretical guarantees of
//! convergence to the global minimum.
//!
//! ## Example
//!
//! ```rust
//! use shgo_rs::{Shgo, ShgoOptions, Bounds};
//!
//! // Rosenbrock function
//! let objective = |x: &[f64]| {
//!     (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
//! };
//!
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//! let options = ShgoOptions {
//!     maxiter: Some(3),
//!     ..Default::default()
//! };
//!
//! let result = Shgo::new(objective, bounds)
//!     .with_options(options)
//!     .minimize()
//!     .unwrap();
//!
//! println!("Global minimum at {:?} with value {}", result.x, result.fun);
//! ```

pub mod vertex;
pub mod coordinates;
pub mod complex;
pub mod sobol;
pub mod error;
pub mod shgo;
pub mod local_opt;
pub mod ffi;

#[cfg(feature = "track-alloc")]
pub mod alloc_tracker;

// Re-export main types
pub use coordinates::Coordinates;
pub use vertex::{Vertex, VertexCache};
pub use complex::Complex;
pub use sobol::Sobol;
pub use error::{ShgoError, Result as ShgoResult};
pub use shgo::{Shgo, ShgoOptions, ShgoResult as OptimizeResult, SamplingMethod, ConnectivityMethod, Bounds, LMapCache, LocalMinimum};
pub use local_opt::{LocalOptimizer, LocalOptimizerOptions, LocalOptResult, minimize_local};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::coordinates::Coordinates;
    pub use crate::vertex::{Vertex, VertexCache};
    pub use crate::complex::Complex;
    pub use crate::sobol::Sobol;
    pub use crate::error::{ShgoError, Result as ShgoResult};
    pub use crate::shgo::{Shgo, ShgoOptions, ShgoResult as OptimizeResult, SamplingMethod, Bounds};
}
