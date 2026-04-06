//! Coordinate types for vertex positions in the simplicial complex.
//!
//! The [`Coordinates`] type provides an immutable representation of a point
//! in n-dimensional space that can be efficiently hashed and compared for
//! use as keys in the vertex cache.
//!
//! Uses [`SmallVec`] internally to avoid heap allocation for dimensions ≤ 8.

use std::fmt;
use std::hash::{Hash, Hasher};

use smallvec::SmallVec;

/// Immutable coordinates representing a point in n-dimensional space.
///
/// Coordinates are designed to be used as keys in hash maps. They implement
/// [`Hash`] and [`Eq`] based on the bit representation of the floating-point
/// values, ensuring consistent hashing behavior.
///
/// Internally uses [`SmallVec<[f64; 8]>`] to store values inline for
/// dimensions ≤ 8, avoiding heap allocation for common problem sizes.
///
/// # Example
///
/// ```
/// use shgo_rs::Coordinates;
///
/// let coords = Coordinates::new(vec![0.5, 1.0, -0.25]);
/// assert_eq!(coords.dim(), 3);
/// assert_eq!(coords[0], 0.5);
/// ```
#[derive(Clone, PartialOrd)]
pub struct Coordinates {
    /// The coordinate values (inline for dim ≤ 8).
    values: SmallVec<[f64; 8]>,
    /// Pre-computed hash for O(1) hash lookups.
    cached_hash: u64,
}

impl Coordinates {
    /// Create new coordinates from a vector of values.
    ///
    /// # Arguments
    ///
    /// * `values` - The coordinate values in each dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Coordinates;
    ///
    /// let coords = Coordinates::new(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(coords.dim(), 3);
    /// ```
    pub fn new(values: Vec<f64>) -> Self {
        let sv: SmallVec<[f64; 8]> = SmallVec::from_vec(values);
        let cached_hash = Self::compute_hash(&sv);
        Self { values: sv, cached_hash }
    }

    /// Create coordinates from a slice.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Coordinates;
    ///
    /// let arr = [1.0, 2.0, 3.0];
    /// let coords = Coordinates::from_slice(&arr);
    /// ```
    pub fn from_slice(values: &[f64]) -> Self {
        Self::new(values.to_vec())
    }

    /// Create zero coordinates of the given dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Coordinates;
    ///
    /// let origin = Coordinates::zeros(3);
    /// assert_eq!(origin.as_slice(), &[0.0, 0.0, 0.0]);
    /// ```
    pub fn zeros(dim: usize) -> Self {
        Self::new(vec![0.0; dim])
    }

    /// Create coordinates filled with a single value.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Coordinates;
    ///
    /// let ones = Coordinates::filled(3, 1.0);
    /// assert_eq!(ones.as_slice(), &[1.0, 1.0, 1.0]);
    /// ```
    pub fn filled(dim: usize, value: f64) -> Self {
        Self::new(vec![value; dim])
    }

    /// Get the dimensionality of the coordinates.
    #[inline]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Get the coordinate values as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Get the coordinate values as a mutable slice.
    /// 
    /// **Warning:** Modifying coordinates after creation will invalidate
    /// the cached hash. Use [`Coordinates::rehash`] after modification.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Convert to a Vec, consuming the coordinates.
    #[inline]
    pub fn into_vec(self) -> Vec<f64> {
        self.values.into_vec()
    }

    /// Recompute the cached hash after modification.
    ///
    /// This must be called after using [`Coordinates::as_mut_slice`] to
    /// modify values, otherwise hash-based lookups will fail.
    pub fn rehash(&mut self) {
        self.cached_hash = Self::compute_hash(&self.values);
    }

    /// Compute the hash of coordinate values.
    ///
    /// Uses the bit representation of f64 values for deterministic hashing.
    fn compute_hash(values: &[f64]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for &v in values {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Compute the midpoint between two coordinates.
    ///
    /// # Panics
    ///
    /// Panics if the coordinates have different dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Coordinates;
    ///
    /// let a = Coordinates::new(vec![0.0, 0.0]);
    /// let b = Coordinates::new(vec![2.0, 4.0]);
    /// let mid = a.midpoint(&b);
    /// assert_eq!(mid.as_slice(), &[1.0, 2.0]);
    /// ```
    pub fn midpoint(&self, other: &Coordinates) -> Coordinates {
        assert_eq!(
            self.dim(),
            other.dim(),
            "Cannot compute midpoint of coordinates with different dimensions"
        );
        let values: SmallVec<[f64; 8]> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();
        let cached_hash = Self::compute_hash(&values);
        Coordinates { values, cached_hash }
    }

    /// Compute the Euclidean distance to another coordinate.
    ///
    /// # Panics
    ///
    /// Panics if the coordinates have different dimensions.
    pub fn distance(&self, other: &Coordinates) -> f64 {
        assert_eq!(
            self.dim(),
            other.dim(),
            "Cannot compute distance between coordinates with different dimensions"
        );
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute the squared Euclidean distance (avoids sqrt).
    pub fn distance_squared(&self, other: &Coordinates) -> f64 {
        assert_eq!(self.dim(), other.dim());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }

    /// Check if coordinates are within bounds.
    ///
    /// # Arguments
    ///
    /// * `bounds` - Slice of (lower, upper) bound pairs for each dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Coordinates;
    ///
    /// let coords = Coordinates::new(vec![0.5, 0.5]);
    /// assert!(coords.in_bounds(&[(0.0, 1.0), (0.0, 1.0)]));
    /// assert!(!coords.in_bounds(&[(0.0, 0.4), (0.0, 1.0)]));
    /// ```
    pub fn in_bounds(&self, bounds: &[(f64, f64)]) -> bool {
        if bounds.len() != self.dim() {
            return false;
        }
        self.values
            .iter()
            .zip(bounds.iter())
            .all(|(&v, &(lo, hi))| v >= lo && v <= hi)
    }
}

impl Hash for Coordinates {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use pre-computed hash for speed
        state.write_u64(self.cached_hash);
    }
}

impl PartialEq for Coordinates {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: check cached hash first
        if self.cached_hash != other.cached_hash {
            return false;
        }
        // Compare bit representations for exact equality
        if self.values.len() != other.values.len() {
            return false;
        }
        self.values
            .iter()
            .zip(other.values.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for Coordinates {}

impl std::ops::Index<usize> for Coordinates {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl fmt::Debug for Coordinates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coordinates({:?})", self.values)
    }
}

impl fmt::Display for Coordinates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.6}", v)?;
        }
        write!(f, ")")
    }
}

impl From<Vec<f64>> for Coordinates {
    fn from(values: Vec<f64>) -> Self {
        Self::new(values)
    }
}

impl From<&[f64]> for Coordinates {
    fn from(values: &[f64]) -> Self {
        Self::from_slice(values)
    }
}

impl<const N: usize> From<[f64; N]> for Coordinates {
    fn from(values: [f64; N]) -> Self {
        let sv: SmallVec<[f64; 8]> = SmallVec::from_slice(&values);
        let cached_hash = Self::compute_hash(&sv);
        Self { values: sv, cached_hash }
    }
}

impl AsRef<[f64]> for Coordinates {
    fn as_ref(&self) -> &[f64] {
        self.values.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_new_coordinates() {
        let coords = Coordinates::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(coords.dim(), 3);
        assert_eq!(coords[0], 1.0);
        assert_eq!(coords[1], 2.0);
        assert_eq!(coords[2], 3.0);
    }

    #[test]
    fn test_zeros() {
        let coords = Coordinates::zeros(4);
        assert_eq!(coords.dim(), 4);
        assert!(coords.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_filled() {
        let coords = Coordinates::filled(3, 5.0);
        assert_eq!(coords.as_slice(), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_equality() {
        let a = Coordinates::new(vec![1.0, 2.0, 3.0]);
        let b = Coordinates::new(vec![1.0, 2.0, 3.0]);
        let c = Coordinates::new(vec![1.0, 2.0, 3.1]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_hash_consistency() {
        let a = Coordinates::new(vec![1.0, 2.0, 3.0]);
        let b = Coordinates::new(vec![1.0, 2.0, 3.0]);

        let mut map = HashMap::new();
        map.insert(a.clone(), "first");
        
        // Should find the same entry with equal coordinates
        assert_eq!(map.get(&b), Some(&"first"));
    }

    #[test]
    fn test_hash_map_usage() {
        let mut map: HashMap<Coordinates, i32> = HashMap::new();
        
        map.insert(Coordinates::new(vec![0.0, 0.0]), 1);
        map.insert(Coordinates::new(vec![1.0, 0.0]), 2);
        map.insert(Coordinates::new(vec![0.0, 1.0]), 3);
        map.insert(Coordinates::new(vec![1.0, 1.0]), 4);

        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&Coordinates::new(vec![0.0, 0.0])), Some(&1));
        assert_eq!(map.get(&Coordinates::new(vec![1.0, 1.0])), Some(&4));
        assert_eq!(map.get(&Coordinates::new(vec![0.5, 0.5])), None);
    }

    #[test]
    fn test_midpoint() {
        let a = Coordinates::new(vec![0.0, 0.0, 0.0]);
        let b = Coordinates::new(vec![2.0, 4.0, 6.0]);
        let mid = a.midpoint(&b);

        assert_eq!(mid.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_distance() {
        let a = Coordinates::new(vec![0.0, 0.0]);
        let b = Coordinates::new(vec![3.0, 4.0]);

        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_in_bounds() {
        let coords = Coordinates::new(vec![0.5, 0.5]);
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        assert!(coords.in_bounds(&bounds));

        let out_of_bounds = Coordinates::new(vec![1.5, 0.5]);
        assert!(!out_of_bounds.in_bounds(&bounds));
    }

    #[test]
    fn test_from_conversions() {
        let from_vec: Coordinates = vec![1.0, 2.0].into();
        assert_eq!(from_vec.dim(), 2);

        let from_array: Coordinates = [1.0, 2.0, 3.0].into();
        assert_eq!(from_array.dim(), 3);

        let slice = &[1.0, 2.0, 3.0, 4.0][..];
        let from_slice: Coordinates = slice.into();
        assert_eq!(from_slice.dim(), 4);
    }

    #[test]
    fn test_display() {
        let coords = Coordinates::new(vec![1.0, 2.5, 3.0]);
        let display = format!("{}", coords);
        assert!(display.contains("1.0"));
        assert!(display.contains("2.5"));
        assert!(display.contains("3.0"));
    }

    #[test]
    fn test_nan_handling() {
        // NaN values should hash consistently (to their bit representation)
        let a = Coordinates::new(vec![f64::NAN, 1.0]);
        let b = Coordinates::new(vec![f64::NAN, 1.0]);
        
        // NaN bit representations are equal
        assert_eq!(a, b);
    }

    #[test]
    fn test_zero_vs_negative_zero() {
        // 0.0 and -0.0 have different bit representations
        let pos_zero = Coordinates::new(vec![0.0]);
        let neg_zero = Coordinates::new(vec![-0.0]);
        
        // They should be considered different for hashing purposes
        // (this matches Python's behavior with float bit hashing)
        assert_ne!(pos_zero, neg_zero);
    }
}
