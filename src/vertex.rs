//! Vertex types and caching for the simplicial complex.
//!
//! This module provides the core data structures for representing vertices
//! in the simplicial complex:
//!
//! - [`Vertex`]: A single vertex with coordinates, neighbors, and field values.
//! - [`VertexCache`]: A thread-safe cache for managing vertices with batch
//!   evaluation of objective functions and constraints.
//!
//! # Thread Safety
//!
//! All types in this module are designed for concurrent access:
//! - Individual vertex fields use fine-grained [`RwLock`]s
//! - The cache uses [`RwLock`] with double-checked locking for efficient reads
//! - Atomic operations are used for simple counters and flags
//!
//! # Parallel Evaluation
//!
//! The [`VertexCache`] supports batch evaluation of objective functions and
//! constraints using [`rayon`] for data parallelism:
//!
//! ```rust,ignore
//! let cache = VertexCache::new(objective_fn, Some(vec![constraint_fn]));
//!
//! // Create vertices (adds them to pending pools)
//! cache.get_or_create(vec![0.0, 0.0]);
//! cache.get_or_create(vec![1.0, 0.0]);
//! cache.get_or_create(vec![0.0, 1.0]);
//!
//! // Evaluate all pending vertices in parallel
//! cache.process_pools();
//! ```

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::coordinates::Coordinates;

/// A vertex in the simplicial complex.
///
/// Each vertex has:
/// - Coordinates (immutable position in n-dimensional space)
/// - A unique index for identification
/// - A set of neighboring vertex indices (graph edges)
/// - Optional field value (objective function evaluation)
/// - Optional feasibility flag (constraint satisfaction)
///
/// # Thread Safety
///
/// Vertex fields that may be modified use interior mutability via [`RwLock`].
/// The `check_min` and `check_max` flags use [`AtomicBool`] for lock-free access.
///
/// # Example
///
/// ```
/// use shgo_rs::Vertex;
///
/// let v = Vertex::new(vec![0.5, 0.5], 0);
/// assert_eq!(v.index(), 0);
/// assert_eq!(v.dim(), 2);
/// ```
pub struct Vertex {
    /// The coordinates of this vertex (immutable).
    coordinates: Coordinates,

    /// Unique index of this vertex in the cache.
    index: usize,

    /// Set of indices of neighboring vertices.
    /// Using indices rather than references avoids lifetime complexity.
    neighbors: RwLock<HashSet<usize>>,

    /// The objective function value at this vertex.
    /// `None` if not yet evaluated, `Some(f64::INFINITY)` if infeasible.
    f: RwLock<Option<f64>>,

    /// Whether this vertex satisfies all constraints.
    /// `None` if not yet checked.
    feasible: RwLock<Option<bool>>,

    /// Flag indicating that minimizer status needs to be recomputed.
    check_min: AtomicBool,

    /// Flag indicating that maximizer status needs to be recomputed.
    check_max: AtomicBool,

    /// Cached minimizer status (valid only when check_min is false).
    cached_min: RwLock<Option<bool>>,

    /// Cached maximizer status (valid only when check_max is false).
    cached_max: RwLock<Option<bool>>,
}

impl Vertex {
    /// Create a new vertex at the given coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - The position in n-dimensional space.
    /// * `index` - The unique index for this vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Vertex;
    ///
    /// let v = Vertex::new(vec![1.0, 2.0, 3.0], 42);
    /// assert_eq!(v.index(), 42);
    /// assert_eq!(v.dim(), 3);
    /// ```
    pub fn new(coords: Vec<f64>, index: usize) -> Self {
        Self {
            coordinates: Coordinates::new(coords),
            index,
            neighbors: RwLock::new(HashSet::new()),
            f: RwLock::new(None),
            feasible: RwLock::new(None),
            check_min: AtomicBool::new(true),
            check_max: AtomicBool::new(true),
            cached_min: RwLock::new(None),
            cached_max: RwLock::new(None),
        }
    }

    /// Create a vertex from existing coordinates.
    pub fn from_coordinates(coordinates: Coordinates, index: usize) -> Self {
        Self {
            coordinates,
            index,
            neighbors: RwLock::new(HashSet::new()),
            f: RwLock::new(None),
            feasible: RwLock::new(None),
            check_min: AtomicBool::new(true),
            check_max: AtomicBool::new(true),
            cached_min: RwLock::new(None),
            cached_max: RwLock::new(None),
        }
    }

    /// Get the unique index of this vertex.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the dimensionality of this vertex's coordinates.
    #[inline]
    pub fn dim(&self) -> usize {
        self.coordinates.dim()
    }

    /// Get a reference to the coordinates.
    #[inline]
    pub fn coordinates(&self) -> &Coordinates {
        &self.coordinates
    }

    /// Get the coordinate values as a slice.
    #[inline]
    pub fn x(&self) -> &[f64] {
        self.coordinates.as_slice()
    }

    /// Get the objective function value, if evaluated.
    pub fn f(&self) -> Option<f64> {
        *self.f.read()
    }

    /// Set the objective function value.
    pub fn set_f(&self, value: f64) {
        *self.f.write() = Some(value);
    }

    /// Get the feasibility status, if checked.
    pub fn feasible(&self) -> Option<bool> {
        *self.feasible.read()
    }

    /// Set the feasibility status.
    pub fn set_feasible(&self, value: bool) {
        *self.feasible.write() = Some(value);
    }

    /// Check if this vertex has been evaluated (has an f value).
    pub fn is_evaluated(&self) -> bool {
        self.f.read().is_some()
    }

    /// Get the set of neighbor indices.
    pub fn neighbor_indices(&self) -> Vec<usize> {
        self.neighbors.read().iter().copied().collect()
    }

    /// Get the number of neighbors.
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.read().len()
    }

    /// Check if a vertex index is a neighbor.
    pub fn has_neighbor(&self, index: usize) -> bool {
        self.neighbors.read().contains(&index)
    }

    /// Connect this vertex to another vertex (by index).
    ///
    /// This only adds the neighbor to this vertex's set. For bidirectional
    /// connection, call this method on both vertices or use
    /// [`Vertex::connect_bidirectional`].
    ///
    /// Invalidates cached minimizer/maximizer status.
    pub fn connect(&self, other_index: usize) {
        if other_index == self.index {
            return; // Don't connect to self
        }
        let mut neighbors = self.neighbors.write();
        if neighbors.insert(other_index) {
            // Connection added, invalidate cached status
            self.check_min.store(true, Ordering::Release);
            self.check_max.store(true, Ordering::Release);
        }
    }

    /// Disconnect from another vertex (by index).
    ///
    /// Invalidates cached minimizer/maximizer status.
    pub fn disconnect(&self, other_index: usize) {
        let mut neighbors = self.neighbors.write();
        if neighbors.remove(&other_index) {
            // Connection removed, invalidate cached status
            self.check_min.store(true, Ordering::Release);
            self.check_max.store(true, Ordering::Release);
        }
    }

    /// Connect two vertices bidirectionally.
    ///
    /// This is a convenience method that connects both vertices to each other.
    pub fn connect_bidirectional(v1: &Vertex, v2: &Vertex) {
        v1.connect(v2.index);
        v2.connect(v1.index);
    }

    /// Disconnect two vertices bidirectionally.
    pub fn disconnect_bidirectional(v1: &Vertex, v2: &Vertex) {
        v1.disconnect(v2.index);
        v2.disconnect(v1.index);
    }

    /// Mark that minimizer/maximizer status needs recomputation.
    pub fn invalidate_extrema_cache(&self) {
        self.check_min.store(true, Ordering::Release);
        self.check_max.store(true, Ordering::Release);
    }

    /// Check if the minimizer status needs recomputation.
    pub fn needs_min_check(&self) -> bool {
        self.check_min.load(Ordering::Acquire)
    }

    /// Check if the maximizer status needs recomputation.
    pub fn needs_max_check(&self) -> bool {
        self.check_max.load(Ordering::Acquire)
    }

    /// Get the cached minimizer status without recomputing.
    pub fn cached_minimizer(&self) -> Option<bool> {
        if self.check_min.load(Ordering::Acquire) {
            None
        } else {
            *self.cached_min.read()
        }
    }

    /// Set the cached minimizer status.
    pub fn set_cached_minimizer(&self, is_min: bool) {
        *self.cached_min.write() = Some(is_min);
        self.check_min.store(false, Ordering::Release);
    }

    /// Get the cached maximizer status without recomputing.
    pub fn cached_maximizer(&self) -> Option<bool> {
        if self.check_max.load(Ordering::Acquire) {
            None
        } else {
            *self.cached_max.read()
        }
    }

    /// Set the cached maximizer status.
    pub fn set_cached_maximizer(&self, is_max: bool) {
        *self.cached_max.write() = Some(is_max);
        self.check_max.store(false, Ordering::Release);
    }
}

impl std::fmt::Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vertex")
            .field("index", &self.index)
            .field("coordinates", &self.coordinates)
            .field("f", &self.f())
            .field("feasible", &self.feasible())
            .field("neighbors", &self.neighbor_count())
            .finish()
    }
}

// Vertex is Send + Sync because all mutable fields use interior mutability
// with thread-safe primitives (RwLock, AtomicBool)
unsafe impl Send for Vertex {}
unsafe impl Sync for Vertex {}

/// A thread-safe cache for managing vertices in the simplicial complex.
///
/// The cache provides:
/// - O(1) lookup of vertices by coordinates
/// - Automatic creation of new vertices on access
/// - Batch evaluation of objective functions and constraints
/// - Parallel processing via rayon
///
/// # Type Parameters
///
/// * `F` - The objective function type: `Fn(&[f64]) -> f64`
/// * `G` - The constraint function type: `Fn(&[f64]) -> bool`
///
/// # Example
///
/// ```rust,ignore
/// let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
/// let constraint = |x: &[f64]| x[0] + x[1] <= 1.0;
///
/// let cache = VertexCache::new(objective, Some(vec![constraint]));
///
/// // Get or create vertices
/// let v1 = cache.get_or_create(vec![0.0, 0.0]);
/// let v2 = cache.get_or_create(vec![1.0, 0.0]);
///
/// // Evaluate all pending vertices in parallel
/// cache.process_pools();
/// ```
pub struct VertexCache<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> bool + Send + Sync,
{
    /// The vertex cache: coordinates -> vertex
    cache: RwLock<IndexMap<Coordinates, Arc<Vertex>>>,

    /// The objective function to evaluate.
    field: F,

    /// Optional constraint functions.
    constraints: Option<Vec<G>>,

    /// Indices of vertices pending field evaluation.
    pending_field: RwLock<Vec<usize>>,

    /// Indices of vertices pending constraint evaluation.
    pending_constraints: RwLock<Vec<usize>>,

    /// Number of feasible function evaluations performed.
    pub nfev: AtomicUsize,

    /// Next available vertex index.
    next_index: AtomicUsize,
}

impl<F, G> VertexCache<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> bool + Send + Sync,
{
    /// Create a new vertex cache with the given objective function.
    ///
    /// # Arguments
    ///
    /// * `field` - The objective function to minimize.
    /// * `constraints` - Optional constraint functions. Each should return
    ///   `true` if the constraint is satisfied.
    pub fn new(field: F, constraints: Option<Vec<G>>) -> Self {
        Self {
            cache: RwLock::new(IndexMap::new()),
            field,
            constraints,
            pending_field: RwLock::new(Vec::new()),
            pending_constraints: RwLock::new(Vec::new()),
            nfev: AtomicUsize::new(0),
            next_index: AtomicUsize::new(0),
        }
    }

    /// Get the number of vertices in the cache.
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Get the number of function evaluations performed.
    pub fn function_evaluations(&self) -> usize {
        self.nfev.load(Ordering::Relaxed)
    }

    /// Get a vertex by coordinates, creating it if it doesn't exist.
    ///
    /// New vertices are automatically added to the pending evaluation pools.
    ///
    /// # Arguments
    ///
    /// * `coords` - The coordinate values.
    ///
    /// # Returns
    ///
    /// An `Arc<Vertex>` for the vertex at the given coordinates.
    pub fn get_or_create(&self, coords: Vec<f64>) -> Arc<Vertex> {
        let key = Coordinates::new(coords.clone());

        // Fast path: check if already exists (read lock only)
        {
            let cache = self.cache.read();
            if let Some(vertex) = cache.get(&key) {
                return vertex.clone();
            }
        }

        // Slow path: need to create new vertex (write lock)
        let mut cache = self.cache.write();

        // Double-check after acquiring write lock (another thread might have added it)
        if let Some(vertex) = cache.get(&key) {
            return vertex.clone();
        }

        // Create new vertex
        let index = self.next_index.fetch_add(1, Ordering::SeqCst);
        let vertex = Arc::new(Vertex::new(coords, index));
        cache.insert(key, vertex.clone());

        // Add to pending pools
        self.pending_field.write().push(index);
        if self.constraints.is_some() {
            self.pending_constraints.write().push(index);
        }

        vertex
    }

    /// Get a vertex by coordinates without creating it.
    ///
    /// # Returns
    ///
    /// `Some(Arc<Vertex>)` if the vertex exists, `None` otherwise.
    pub fn get(&self, coords: &Coordinates) -> Option<Arc<Vertex>> {
        self.cache.read().get(coords).cloned()
    }

    /// Get a vertex by its index.
    ///
    /// This is O(n) as it must search all vertices.
    pub fn get_by_index(&self, index: usize) -> Option<Arc<Vertex>> {
        self.cache.read().values().find(|v| v.index() == index).cloned()
    }

    /// Iterate over all vertices in the cache.
    pub fn iter(&self) -> impl Iterator<Item = Arc<Vertex>> + '_ {
        // Clone the values to avoid holding the lock
        let vertices: Vec<_> = self.cache.read().values().cloned().collect();
        vertices.into_iter()
    }

    /// Process all pending constraint evaluations.
    ///
    /// Vertices that fail any constraint will have their `feasible` field set
    /// to `false`.
    pub fn process_constraints(&self) {
        if self.constraints.is_none() {
            return;
        }

        let pending: Vec<usize> = std::mem::take(&mut *self.pending_constraints.write());
        if pending.is_empty() {
            return;
        }

        let constraints = self.constraints.as_ref().unwrap();
        let cache = self.cache.read();

        // Build a vec of (vertex, coords) for parallel processing
        let vertices: Vec<_> = pending
            .iter()
            .filter_map(|&idx| {
                cache.values().find(|v| v.index() == idx).cloned()
            })
            .collect();

        // Process constraints in parallel
        vertices.par_iter().for_each(|vertex| {
            let x = vertex.x();
            let feasible = constraints.iter().all(|g| g(x));
            vertex.set_feasible(feasible);
            if !feasible {
                vertex.set_f(f64::INFINITY);
            }
        });
    }

    /// Process all pending field evaluations.
    ///
    /// Only evaluates vertices that are feasible (or have no constraints).
    /// Infeasible vertices are assigned `f64::INFINITY`.
    pub fn process_field(&self) {
        let pending: Vec<usize> = std::mem::take(&mut *self.pending_field.write());
        if pending.is_empty() {
            return;
        }

        let cache = self.cache.read();

        // Build vec of vertices to evaluate
        let vertices: Vec<_> = pending
            .iter()
            .filter_map(|&idx| {
                cache.values().find(|v| v.index() == idx).cloned()
            })
            .collect();

        // Process field evaluations in parallel
        let eval_count = AtomicUsize::new(0);

        vertices.par_iter().for_each(|vertex| {
            // Skip if already evaluated
            if vertex.is_evaluated() {
                return;
            }

            // Check feasibility
            let is_feasible = vertex.feasible().unwrap_or(true);

            if is_feasible {
                let f_val = (self.field)(vertex.x());
                let f_val = if f_val.is_nan() { f64::INFINITY } else { f_val };
                vertex.set_f(f_val);
                eval_count.fetch_add(1, Ordering::Relaxed);
            } else {
                vertex.set_f(f64::INFINITY);
            }
        });

        self.nfev.fetch_add(eval_count.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Process all pending evaluations (constraints first, then field).
    ///
    /// This is the main entry point for batch evaluation of vertices.
    pub fn process_pools(&self) {
        self.process_constraints();
        self.process_field();
    }

    /// Check if a vertex is a local minimizer.
    ///
    /// A vertex is a minimizer if its function value is strictly less than
    /// all of its neighbors' function values.
    ///
    /// # Arguments
    ///
    /// * `vertex` - The vertex to check.
    ///
    /// # Returns
    ///
    /// `true` if the vertex is a local minimizer, `false` otherwise.
    /// Returns `false` if the vertex has not been evaluated or is infeasible.
    pub fn is_minimizer(&self, vertex: &Vertex) -> bool {
        // Check cached value first
        if let Some(is_min) = vertex.cached_minimizer() {
            return is_min;
        }

        // Get this vertex's function value
        let f = match vertex.f() {
            Some(f) if f.is_finite() => f,
            _ => {
                vertex.set_cached_minimizer(false);
                return false;
            }
        };

        // Check all neighbors
        let neighbor_indices = vertex.neighbor_indices();
        let cache = self.cache.read();

        let is_min = neighbor_indices.iter().all(|&nn_idx| {
            if let Some(nn) = cache.values().find(|v| v.index() == nn_idx) {
                match nn.f() {
                    Some(nn_f) => f < nn_f,
                    None => true, // Unevaluated neighbor doesn't disqualify
                }
            } else {
                true // Missing neighbor doesn't disqualify
            }
        });

        vertex.set_cached_minimizer(is_min);
        is_min
    }

    /// Check if a vertex is a local maximizer.
    pub fn is_maximizer(&self, vertex: &Vertex) -> bool {
        if let Some(is_max) = vertex.cached_maximizer() {
            return is_max;
        }

        let f = match vertex.f() {
            Some(f) if f.is_finite() => f,
            _ => {
                vertex.set_cached_maximizer(false);
                return false;
            }
        };

        let neighbor_indices = vertex.neighbor_indices();
        let cache = self.cache.read();

        let is_max = neighbor_indices.iter().all(|&nn_idx| {
            if let Some(nn) = cache.values().find(|v| v.index() == nn_idx) {
                match nn.f() {
                    Some(nn_f) => f > nn_f,
                    None => true,
                }
            } else {
                true
            }
        });

        vertex.set_cached_maximizer(is_max);
        is_max
    }

    /// Find all local minimizers in the cache.
    ///
    /// This processes minimizer checks in parallel.
    pub fn find_all_minimizers(&self) -> Vec<Arc<Vertex>> {
        let vertices: Vec<_> = self.cache.read().values().cloned().collect();

        vertices
            .into_par_iter()
            .filter(|v| self.is_minimizer(v))
            .collect()
    }

    /// Find all local maximizers in the cache.
    pub fn find_all_maximizers(&self) -> Vec<Arc<Vertex>> {
        let vertices: Vec<_> = self.cache.read().values().cloned().collect();

        vertices
            .into_par_iter()
            .filter(|v| self.is_maximizer(v))
            .collect()
    }

    /// Update minimizer/maximizer status for all vertices.
    pub fn update_extrema(&self) {
        let vertices: Vec<_> = self.cache.read().values().cloned().collect();

        vertices.par_iter().for_each(|v| {
            self.is_minimizer(v);
            self.is_maximizer(v);
        });
    }
}

impl<F, G> std::fmt::Debug for VertexCache<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> bool + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexCache")
            .field("size", &self.len())
            .field("nfev", &self.function_evaluations())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_creation() {
        let v = Vertex::new(vec![1.0, 2.0, 3.0], 0);
        assert_eq!(v.index(), 0);
        assert_eq!(v.dim(), 3);
        assert_eq!(v.x(), &[1.0, 2.0, 3.0]);
        assert!(v.f().is_none());
        assert!(v.feasible().is_none());
    }

    #[test]
    fn test_vertex_field_setting() {
        let v = Vertex::new(vec![0.0, 0.0], 0);
        assert!(v.f().is_none());

        v.set_f(42.0);
        assert_eq!(v.f(), Some(42.0));

        v.set_feasible(true);
        assert_eq!(v.feasible(), Some(true));
    }

    #[test]
    fn test_vertex_connections() {
        let v1 = Vertex::new(vec![0.0, 0.0], 0);
        let v2 = Vertex::new(vec![1.0, 0.0], 1);
        let v3 = Vertex::new(vec![0.0, 1.0], 2);

        // Connect v1 to v2 and v3
        v1.connect(v2.index());
        v1.connect(v3.index());

        assert_eq!(v1.neighbor_count(), 2);
        assert!(v1.has_neighbor(1));
        assert!(v1.has_neighbor(2));
        assert!(!v1.has_neighbor(0)); // Not connected to self

        // Disconnect v1 from v2
        v1.disconnect(v2.index());
        assert_eq!(v1.neighbor_count(), 1);
        assert!(!v1.has_neighbor(1));
        assert!(v1.has_neighbor(2));
    }

    #[test]
    fn test_vertex_bidirectional_connection() {
        let v1 = Vertex::new(vec![0.0, 0.0], 0);
        let v2 = Vertex::new(vec![1.0, 1.0], 1);

        Vertex::connect_bidirectional(&v1, &v2);

        assert!(v1.has_neighbor(v2.index()));
        assert!(v2.has_neighbor(v1.index()));
    }

    #[test]
    fn test_vertex_self_connection_prevented() {
        let v = Vertex::new(vec![0.0, 0.0], 0);
        v.connect(v.index());
        assert_eq!(v.neighbor_count(), 0);
    }

    #[test]
    fn test_cache_creation() {
        let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        assert!(cache.is_empty());
        assert_eq!(cache.function_evaluations(), 0);
    }

    #[test]
    fn test_cache_get_or_create() {
        let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        let v1 = cache.get_or_create(vec![0.0, 0.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(v1.index(), 0);

        // Same coordinates should return same vertex
        let v1_again = cache.get_or_create(vec![0.0, 0.0]);
        assert_eq!(v1.index(), v1_again.index());
        assert_eq!(cache.len(), 1);

        // Different coordinates should create new vertex
        let v2 = cache.get_or_create(vec![1.0, 0.0]);
        assert_eq!(cache.len(), 2);
        assert_eq!(v2.index(), 1);
    }

    #[test]
    fn test_cache_process_field() {
        let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        let v1 = cache.get_or_create(vec![0.0, 0.0]);
        let v2 = cache.get_or_create(vec![1.0, 0.0]);
        let v3 = cache.get_or_create(vec![3.0, 4.0]);

        // Before processing, no f values
        assert!(v1.f().is_none());
        assert!(v2.f().is_none());
        assert!(v3.f().is_none());

        // Process evaluations
        cache.process_pools();

        // After processing, f values should be set
        assert_eq!(v1.f(), Some(0.0));  // 0^2 + 0^2 = 0
        assert_eq!(v2.f(), Some(1.0));  // 1^2 + 0^2 = 1
        assert_eq!(v3.f(), Some(25.0)); // 3^2 + 4^2 = 25

        assert_eq!(cache.function_evaluations(), 3);
    }

    #[test]
    fn test_cache_with_constraints() {
        let objective = |x: &[f64]| x[0] + x[1];
        // Constraint: x[0] >= 0 AND x[1] >= 0
        let constraint = |x: &[f64]| x[0] >= 0.0 && x[1] >= 0.0;

        let cache = VertexCache::new(objective, Some(vec![constraint]));

        let v_feasible = cache.get_or_create(vec![1.0, 1.0]);
        let v_infeasible = cache.get_or_create(vec![-1.0, 1.0]);

        cache.process_pools();

        // Feasible vertex should have f value
        assert_eq!(v_feasible.feasible(), Some(true));
        assert_eq!(v_feasible.f(), Some(2.0));

        // Infeasible vertex should have infinity
        assert_eq!(v_infeasible.feasible(), Some(false));
        assert_eq!(v_infeasible.f(), Some(f64::INFINITY));

        // Only feasible evaluations counted
        assert_eq!(cache.function_evaluations(), 1);
    }

    #[test]
    fn test_minimizer_detection() {
        let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        // Create a simple triangle with origin at center
        let v_center = cache.get_or_create(vec![0.0, 0.0]);  // f = 0 (minimum)
        let v1 = cache.get_or_create(vec![1.0, 0.0]);         // f = 1
        let v2 = cache.get_or_create(vec![0.0, 1.0]);         // f = 1
        let v3 = cache.get_or_create(vec![-1.0, 0.0]);        // f = 1

        // Connect center to all others
        v_center.connect(v1.index());
        v_center.connect(v2.index());
        v_center.connect(v3.index());
        v1.connect(v_center.index());
        v2.connect(v_center.index());
        v3.connect(v_center.index());

        // Evaluate
        cache.process_pools();

        // Center should be the only minimizer
        assert!(cache.is_minimizer(&v_center));
        assert!(!cache.is_minimizer(&v1));
        assert!(!cache.is_minimizer(&v2));
        assert!(!cache.is_minimizer(&v3));
    }

    #[test]
    fn test_find_all_minimizers() {
        let objective = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        // Create vertices
        let v_min = cache.get_or_create(vec![1.0, 1.0]);  // f = 0 (global min)
        let v1 = cache.get_or_create(vec![0.0, 1.0]);     // f = 1
        let v2 = cache.get_or_create(vec![2.0, 1.0]);     // f = 1
        let v3 = cache.get_or_create(vec![1.0, 0.0]);     // f = 1
        let v4 = cache.get_or_create(vec![1.0, 2.0]);     // f = 1

        // Connect v_min to all neighbors
        for v in [&v1, &v2, &v3, &v4] {
            Vertex::connect_bidirectional(&v_min, v);
        }

        cache.process_pools();

        let minimizers = cache.find_all_minimizers();
        assert_eq!(minimizers.len(), 1);
        assert_eq!(minimizers[0].index(), v_min.index());
    }

    #[test]
    fn test_nan_handling() {
        let objective = |x: &[f64]| {
            if x[0] < 0.0 {
                f64::NAN
            } else {
                x[0]
            }
        };
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        let v = cache.get_or_create(vec![-1.0]);
        cache.process_pools();

        // NaN should be converted to infinity
        assert_eq!(v.f(), Some(f64::INFINITY));
    }

    #[test]
    fn test_cache_iterator() {
        let objective = |_x: &[f64]| 0.0;
        let cache: VertexCache<_, fn(&[f64]) -> bool> = VertexCache::new(objective, None);

        cache.get_or_create(vec![0.0]);
        cache.get_or_create(vec![1.0]);
        cache.get_or_create(vec![2.0]);

        let vertices: Vec<_> = cache.iter().collect();
        assert_eq!(vertices.len(), 3);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let objective = |x: &[f64]| x[0].powi(2);
        let cache: Arc<VertexCache<_, fn(&[f64]) -> bool>> =
            Arc::new(VertexCache::new(objective, None));

        let mut handles = vec![];

        // Spawn multiple threads that create vertices
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    cache_clone.get_or_create(vec![i as f64 + j as f64 / 100.0]);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All unique coordinates should be in the cache
        assert_eq!(cache.len(), 1000);
    }
}
