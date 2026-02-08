//! Error types for the SHGO library.

use thiserror::Error;

/// Result type alias for SHGO operations.
pub type Result<T> = std::result::Result<T, ShgoError>;

/// Errors that can occur during SHGO optimization.
#[derive(Error, Debug)]
pub enum ShgoError {
    /// Invalid bounds specification.
    #[error("Invalid bounds: {0}")]
    InvalidBounds(String),

    /// Dimension mismatch between inputs.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// No feasible point found in the search domain.
    #[error("No feasible point found: {0}")]
    NoFeasiblePoint(String),

    /// Optimization failed to converge.
    #[error("Failed to converge: {0}")]
    ConvergenceError(String),

    /// Invalid configuration option.
    #[error("Invalid option: {0}")]
    InvalidOption(String),

    /// Vertex not found in cache.
    #[error("Vertex not found at coordinates: {0:?}")]
    VertexNotFound(Vec<f64>),

    /// Numerical error during computation.
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Mesh/triangulation generation error (e.g., Delaunay failure).
    #[error("Mesh generation error: {0}")]
    MeshGeneration(String),

    /// Optimization was cancelled by user.
    #[error("Optimization cancelled")]
    Cancelled,
}
