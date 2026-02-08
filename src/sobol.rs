//! Sobol sequence generator for quasi-random sampling.
//!
//! This module provides a Sobol sequence generator for generating low-discrepancy
//! quasi-random points in n-dimensional space. These sequences have better uniformity
//! properties than pseudo-random sequences, making them ideal for numerical integration
//! and global optimization.
//!
//! This implementation uses Joe-Kuo (2008) direction numbers via the `sobol-qmc` crate,
//! which matches the direction numbers used by `scipy.stats.qmc.Sobol` in Python.
//! This ensures faithful reproduction of the same sample points as the Python SHGO
//! implementation.
//!
//! # References
//!
//! - Sobol, I. M. (1967). "On the distribution of points in a cube and the approximate
//!   evaluation of integrals". USSR Computational Mathematics and Mathematical Physics.
//! - Joe, S. & Kuo, F. Y. (2008). "Constructing Sobol sequences with better
//!   two-dimensional projections". SIAM J. Sci. Comput. 30, 2635-2654.
//!
//! # Example
//!
//! ```
//! use shgo_rs::Sobol;
//!
//! let mut sobol = Sobol::new(2);
//! let points = sobol.generate(10, 1);
//!
//! assert_eq!(points.len(), 10);
//! for point in &points {
//!     assert_eq!(point.len(), 2);
//!     for &x in point {
//!         assert!(0.0 <= x && x <= 1.0);
//!     }
//! }
//! ```

use sobol_qmc::params::JoeKuoD6;

/// Maximum supported dimensionality (Joe-Kuo supports up to 21,201).
/// We use the STANDARD parameter set (1,000 dims) by default, which is
/// more than sufficient for typical optimization problems.
pub const DIM_MAX: usize = 1000;

/// Sobol sequence generator using Joe-Kuo (2008) direction numbers.
///
/// This generator produces quasi-random points in [0, 1)^dim using the same
/// direction numbers as scipy's `qmc.Sobol(d=dim, scramble=False)`.
pub struct Sobol {
    dim: usize,
    /// The sobol-qmc iterator producing f64 points in [0, 1).
    inner: sobol_qmc::Sobol<f64>,
    /// Current index in the sequence (0-based).
    seed: u64,
}

impl Sobol {
    /// Create a new Sobol sequence generator for the given dimensionality.
    ///
    /// # Arguments
    ///
    /// * `dim` - The number of dimensions (1 to 1000).
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0 or greater than `DIM_MAX`.
    pub fn new(dim: usize) -> Self {
        assert!(
            dim >= 1 && dim <= DIM_MAX,
            "Dimension must be between 1 and {} (got {})",
            DIM_MAX,
            dim
        );

        let params = JoeKuoD6::STANDARD;
        let inner = sobol_qmc::Sobol::<f64>::new(dim, &params)
            .expect("Failed to create Sobol sequence");

        Sobol {
            dim,
            inner,
            seed: 0,
        }
    }

    /// Get the dimensionality of this generator.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the current seed/index.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Reset the generator to a specific seed (sequence index).
    ///
    /// # Arguments
    ///
    /// * `seed` - The new seed value (index in sequence).
    pub fn reset(&mut self, seed: u64) {
        // Re-create the iterator from scratch
        let params = JoeKuoD6::STANDARD;
        self.inner = sobol_qmc::Sobol::<f64>::new(self.dim, &params)
            .expect("Failed to create Sobol sequence");
        self.seed = 0;

        // Skip forward to the desired position
        // The sobol-qmc iterator yields points starting from index 0 (origin)
        for _ in 0..seed {
            let _ = self.inner.next();
            self.seed += 1;
        }
    }

    /// Generate the next point in the sequence.
    ///
    /// # Returns
    ///
    /// A vector of `dim` values, each in [0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Sobol;
    ///
    /// let mut sobol = Sobol::new(2);
    /// let point = sobol.next();
    /// assert_eq!(point.len(), 2);
    /// ```
    pub fn next(&mut self) -> Vec<f64> {
        let point = self.inner.next()
            .expect("Sobol sequence exhausted");
        self.seed += 1;
        point
    }

    /// Generate multiple points from the sequence.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of points to generate.
    /// * `skip` - Number of initial points to skip (resets generator first).
    ///
    /// # Returns
    ///
    /// A vector of `n` points, each a vector of `dim` values in [0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Sobol;
    ///
    /// let mut sobol = Sobol::new(2);
    /// let points = sobol.generate(100, 1);
    /// assert_eq!(points.len(), 100);
    /// ```
    pub fn generate(&mut self, n: usize, skip: usize) -> Vec<Vec<f64>> {
        self.reset(skip as u64);

        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            result.push(self.next());
        }
        result
    }

    /// Generate points scaled to custom bounds.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of points to generate.
    /// * `bounds` - Bounds as (lower, upper) pairs for each dimension.
    /// * `skip` - Number of initial points to skip.
    ///
    /// # Returns
    ///
    /// A vector of `n` points scaled to the given bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Sobol;
    ///
    /// let mut sobol = Sobol::new(2);
    /// let bounds = vec![(-1.0, 1.0), (0.0, 10.0)];
    /// let points = sobol.generate_bounds(10, &bounds, 1);
    ///
    /// for point in &points {
    ///     assert!(point[0] >= -1.0 && point[0] <= 1.0);
    ///     assert!(point[1] >= 0.0 && point[1] <= 10.0);
    /// }
    /// ```
    pub fn generate_bounds(&mut self, n: usize, bounds: &[(f64, f64)], skip: usize) -> Vec<Vec<f64>> {
        assert_eq!(bounds.len(), self.dim, "Bounds must match dimensionality");

        let points = self.generate(n, skip);

        points
            .into_iter()
            .map(|p| {
                p.iter()
                    .zip(bounds.iter())
                    .map(|(&x, &(lo, hi))| lo + x * (hi - lo))
                    .collect()
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobol_creation() {
        let sobol = Sobol::new(3);
        assert_eq!(sobol.dim(), 3);
        assert_eq!(sobol.seed(), 0);
    }

    #[test]
    fn test_sobol_1d() {
        let mut sobol = Sobol::new(1);
        let points = sobol.generate(10, 1);

        assert_eq!(points.len(), 10);
        for point in &points {
            assert_eq!(point.len(), 1);
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
        }
    }

    #[test]
    fn test_sobol_2d() {
        let mut sobol = Sobol::new(2);
        let points = sobol.generate(100, 1);

        assert_eq!(points.len(), 100);
        for point in &points {
            assert_eq!(point.len(), 2);
            for &x in point {
                assert!(x >= 0.0 && x <= 1.0);
            }
        }

        // Check that points are unique
        let mut seen = std::collections::HashSet::new();
        for point in &points {
            let key = format!("{:.10},{:.10}", point[0], point[1]);
            assert!(seen.insert(key), "Duplicate point found");
        }
    }

    #[test]
    fn test_sobol_3d() {
        let mut sobol = Sobol::new(3);
        let points = sobol.generate(50, 1);

        assert_eq!(points.len(), 50);
        for point in &points {
            assert_eq!(point.len(), 3);
        }
    }

    #[test]
    fn test_sobol_bounds() {
        let mut sobol = Sobol::new(2);
        let bounds = vec![(-5.0, 5.0), (0.0, 100.0)];
        let points = sobol.generate_bounds(50, &bounds, 1);

        for point in &points {
            assert!(point[0] >= -5.0 && point[0] <= 5.0);
            assert!(point[1] >= 0.0 && point[1] <= 100.0);
        }
    }

    #[test]
    fn test_sobol_skip() {
        let mut sobol1 = Sobol::new(2);
        let mut sobol2 = Sobol::new(2);

        // Generate with different skips
        let _skip1 = sobol1.generate(5, 0);
        let point1 = sobol1.next();

        sobol2.reset(5);
        let point2 = sobol2.next();

        // Should be the same point
        assert_eq!(point1.len(), point2.len());
        for (a, b) in point1.iter().zip(point2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sobol_reproducibility() {
        let mut sobol1 = Sobol::new(2);
        let mut sobol2 = Sobol::new(2);

        let points1 = sobol1.generate(20, 1);
        let points2 = sobol2.generate(20, 1);

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            for (a, b) in p1.iter().zip(p2.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_sobol_high_dim() {
        // Test high dimensionality (now up to 1000 with Joe-Kuo)
        let mut sobol = Sobol::new(100);
        let points = sobol.generate(10, 1);

        assert_eq!(points.len(), 10);
        for point in &points {
            assert_eq!(point.len(), 100);
        }
    }

    #[test]
    #[should_panic(expected = "Dimension must be between")]
    fn test_sobol_invalid_dim_zero() {
        Sobol::new(0);
    }

    #[test]
    #[should_panic(expected = "Dimension must be between")]
    fn test_sobol_invalid_dim_too_high() {
        Sobol::new(1001);
    }

    #[test]
    fn test_sobol_first_points_2d() {
        // Joe-Kuo Sobol points for 2D: first point is origin, second is (0.5, 0.5)
        let mut sobol = Sobol::new(2);
        sobol.reset(0);

        let p0 = sobol.next();
        assert!((p0[0] - 0.0).abs() < 1e-10);
        assert!((p0[1] - 0.0).abs() < 1e-10);

        let p1 = sobol.next();
        assert!((p1[0] - 0.5).abs() < 1e-10);
        assert!((p1[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sobol_first_points_1d() {
        // 1D Sobol: 0.0, 0.5, 0.75, 0.25, ...
        let mut sobol = Sobol::new(1);
        sobol.reset(0);

        let p0 = sobol.next();
        assert!((p0[0] - 0.0).abs() < 1e-10);

        let p1 = sobol.next();
        assert!((p1[0] - 0.5).abs() < 1e-10);

        let p2 = sobol.next();
        assert!((p2[0] - 0.75).abs() < 1e-10);

        let p3 = sobol.next();
        assert!((p3[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_sobol_joe_kuo_dim2_sequence() {
        // Validate against known Joe-Kuo 2D sequence (same as scipy)
        // scipy.stats.qmc.Sobol(d=2, scramble=False).random(8) produces:
        // [0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75],
        // [0.375, 0.375], [0.875, 0.875], [0.625, 0.125], [0.125, 0.625]
        let mut sobol = Sobol::new(2);
        sobol.reset(0);

        let expected = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.25, 0.75],
            vec![0.375, 0.375],
            vec![0.875, 0.875],
            vec![0.625, 0.125],
            vec![0.125, 0.625],
        ];

        for (i, exp) in expected.iter().enumerate() {
            let point = sobol.next();
            for (j, (&got, &want)) in point.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-10,
                    "Point {}, dim {}: got {}, expected {}",
                    i, j, got, want
                );
            }
        }
    }
}
