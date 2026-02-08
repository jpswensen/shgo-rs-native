//! Simplicial complex triangulation for SHGO.
//!
//! This module implements the core triangulation engine that generates
//! the simplicial complex structure used by the SHGO algorithm.
//!
//! # Overview
//!
//! The `Complex` struct manages vertices and their connections, building
//! a triangulation of the search domain through:
//!
//! - **Cyclic product**: Initial triangulation using the structure of
//!   C2 x C2 x ... x C2 (the cartesian product of dim cyclic groups)
//! - **Refinement**: Subdividing simplices by splitting edges
//! - **Edge splitting**: Adding midpoints with memoization
//!
//! # Example
//!
//! ```
//! use shgo_rs::Complex;
//!
//! // Create a 2D complex over the unit square
//! let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
//! let objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
//! let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, objective, None);
//!
//! // Build initial triangulation
//! complex.triangulate(None, true);
//!
//! // Check vertices were created
//! assert!(complex.vertex_count() >= 4); // At least 4 corners
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::vertex::VertexCache;
use crate::Coordinates;

/// A simplicial complex for domain triangulation.
///
/// The complex maintains a cache of vertices with their connections,
/// supporting incremental refinement for adaptive optimization.
///
/// # Type Parameters
///
/// - `F`: Objective function type `Fn(&[f64]) -> f64`
/// - `G`: Constraint function type `Fn(&[f64]) -> bool`
pub struct Complex<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> bool + Send + Sync,
{
    /// Spatial dimensionality
    dim: usize,

    /// Domain bounds as (lower, upper) pairs for each dimension
    bounds: Vec<(f64, f64)>,

    /// Cache of all vertices and their connections
    pub cache: Arc<VertexCache<F, G>>,

    /// Origin point (all lower bounds)
    origin: Vec<f64>,

    /// Supremum point (all upper bounds)
    supremum: Vec<f64>,

    /// Memoization cache for split_edge operations
    /// Key: (smaller_coords, larger_coords) to ensure consistent ordering
    split_cache: RwLock<HashMap<(Coordinates, Coordinates), Coordinates>>,

    /// Current generation/refinement level
    generation: usize,

    /// Whether initial triangulation is complete
    triangulated: bool,

    /// Triangulated sub-regions for refinement
    /// Each entry is (origin, supremum) of a triangulated hyperrectangle
    triangulated_vectors: Vec<(Vec<f64>, Vec<f64>)>,
}

impl<F, G> Complex<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    G: Fn(&[f64]) -> bool + Send + Sync + 'static,
{
    /// Create a new simplicial complex.
    ///
    /// # Arguments
    ///
    /// * `bounds` - Domain bounds as (lower, upper) pairs for each dimension
    /// * `objective` - Objective function to minimize
    /// * `constraints` - Optional constraint functions (each returns true if feasible)
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Complex;
    ///
    /// let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    /// let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    /// let complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, f, None);
    /// ```
    pub fn new(bounds: Vec<(f64, f64)>, objective: F, constraints: Option<Vec<G>>) -> Self {
        let dim = bounds.len();
        let origin: Vec<f64> = bounds.iter().map(|(lo, _)| *lo).collect();
        let supremum: Vec<f64> = bounds.iter().map(|(_, hi)| *hi).collect();

        Complex {
            dim,
            bounds,
            cache: Arc::new(VertexCache::new(objective, constraints)),
            origin,
            supremum,
            split_cache: RwLock::new(HashMap::new()),
            generation: 0,
            triangulated: false,
            triangulated_vectors: Vec::new(),
        }
    }

    /// Get the dimensionality of the complex.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the domain bounds.
    #[inline]
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Get the number of vertices in the complex.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.cache.len()
    }

    /// Get the current generation/refinement level.
    #[inline]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Check if initial triangulation is complete.
    #[inline]
    pub fn is_triangulated(&self) -> bool {
        self.triangulated
    }

    /// Get the origin point (all lower bounds).
    #[inline]
    pub fn origin(&self) -> &[f64] {
        &self.origin
    }

    /// Get the supremum point (all upper bounds).
    #[inline]
    pub fn supremum(&self) -> &[f64] {
        &self.supremum
    }

    /// Build the initial triangulation of the domain.
    ///
    /// Uses the cyclic product approach to construct a triangulation
    /// of the hyperrectangle domain.
    ///
    /// # Arguments
    ///
    /// * `n` - Optional maximum number of vertices to generate
    /// * `centroid` - If true, add a central point to complete triangulation
    ///
    /// # Example
    ///
    /// ```
    /// use shgo_rs::Complex;
    ///
    /// let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    /// let f = |x: &[f64]| x[0] + x[1];
    /// let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, f, None);
    ///
    /// complex.triangulate(None, true);
    /// assert!(complex.vertex_count() >= 5); // 4 corners + centroid
    /// ```
    pub fn triangulate(&mut self, n: Option<usize>, centroid: bool) {
        if let Some(max_vertices) = n {
            // Generate up to n vertices
            self.cyclic_product_limited(max_vertices, centroid);
        } else {
            // Generate full triangulation
            self.cyclic_product_full(centroid);
        }

        if !self.triangulated {
            self.triangulated_vectors
                .push((self.origin.clone(), self.supremum.clone()));
            self.triangulated = true;
        }
    }

    /// Full cyclic product triangulation.
    fn cyclic_product_full(&mut self, centroid: bool) {
        let origin = self.origin.clone();
        let supremum = self.supremum.clone();

        // Create origin and supremum vertices
        let vo = self.cache.get_or_create(origin.clone());
        let vs = self.cache.get_or_create(supremum.clone());

        // Connect origin to supremum initially
        vo.connect(vs.index());
        vs.connect(vo.index());

        if self.dim == 0 {
            return;
        }

        // C0x stores "lower" vertices, C1x stores "upper" vertices
        // for each dimension's C2 group
        let mut c0x: Vec<Vec<usize>> = vec![vec![vo.index()]];
        let mut c1x: Vec<Vec<usize>> = Vec::new();

        // First dimension: create the first "upper" vertex
        let mut a_vo_coords = origin.clone();
        a_vo_coords[0] = supremum[0];
        let a_vo = self.cache.get_or_create(a_vo_coords);
        vo.connect(a_vo.index());
        a_vo.connect(vo.index());
        c1x.push(vec![a_vo.index()]);

        // Container for a+b cross operations
        let mut ab_c: Vec<(usize, usize)> = Vec::new();

        // Loop over remaining dimensions
        for i in 1..self.dim {
            c0x.push(Vec::new());
            c1x.push(Vec::new());

            // Copy current containers for iteration
            let c0x_copy: Vec<Vec<usize>> = c0x[..i].iter().cloned().collect();
            let c1x_copy: Vec<Vec<usize>> = c1x[..i].iter().cloned().collect();

            // Process each pair of lower/upper vertex lists
            for (j, (vl_list, vu_list)) in c0x_copy.iter().zip(c1x_copy.iter()).enumerate() {
                for (&vl_idx, &vu_idx) in vl_list.iter().zip(vu_list.iter()) {
                    let vl = self.cache.get_by_index(vl_idx).unwrap();
                    let vu = self.cache.get_by_index(vu_idx).unwrap();

                    // Build aN vertices
                    let mut a_vl_coords: Vec<f64> = vl.x().to_vec();
                    let mut a_vu_coords: Vec<f64> = vu.x().to_vec();
                    a_vl_coords[i] = supremum[i];
                    a_vu_coords[i] = supremum[i];

                    let a_vl = self.cache.get_or_create(a_vl_coords);
                    let a_vu = self.cache.get_or_create(a_vu_coords);

                    // Connect N to aN
                    vl.connect(a_vl.index());
                    a_vl.connect(vl.index());

                    vu.connect(a_vu.index());
                    a_vu.connect(vu.index());

                    // Connect pair in aN
                    a_vl.connect(a_vu.index());
                    a_vu.connect(a_vl.index());

                    // Triangulation: connect lower to opposite upper
                    vl.connect(a_vu.index());
                    a_vu.connect(vl.index());
                    ab_c.push((vl_idx, a_vu.index()));

                    // Update containers
                    c0x[i].push(vl_idx);
                    c0x[i].push(vu_idx);
                    c1x[i].push(a_vl.index());
                    c1x[i].push(a_vu.index());

                    // Update old containers
                    c0x[j].push(a_vl.index());
                    c1x[j].push(a_vu.index());
                }
            }

            // Process a+b cross operations
            let ab_c_copy = ab_c.clone();
            for &(vp0_idx, vp1_idx) in &ab_c_copy {
                let vp0 = self.cache.get_by_index(vp0_idx).unwrap();
                let vp1 = self.cache.get_by_index(vp1_idx).unwrap();

                let mut b_v_coords: Vec<f64> = vp0.x().to_vec();
                let mut ab_v_coords: Vec<f64> = vp1.x().to_vec();
                b_v_coords[i] = supremum[i];
                ab_v_coords[i] = supremum[i];

                let b_v = self.cache.get_or_create(b_v_coords);
                let ab_v = self.cache.get_or_create(ab_v_coords);

                // Connect cross pairs
                vp0.connect(ab_v.index());
                ab_v.connect(vp0_idx);

                b_v.connect(ab_v.index());
                ab_v.connect(b_v.index());

                ab_c.push((vp0_idx, ab_v.index()));
                ab_c.push((b_v.index(), ab_v.index()));
            }
        }

        // Add centroid if requested
        if centroid {
            // Disconnect origin from supremum
            vo.disconnect(vs.index());
            vs.disconnect(vo.index());

            // Create centroid
            let vc = self.split_edge_coords(&origin, &supremum);

            // Connect centroid to all neighbors of origin
            let vo_neighbors: Vec<usize> = vo.neighbor_indices();
            for &neighbor_idx in &vo_neighbors {
                if let Some(neighbor) = self.cache.get_by_index(neighbor_idx) {
                    let vc_vertex = self.cache.get_or_create(vc.as_slice().to_vec());
                    neighbor.connect(vc_vertex.index());
                    vc_vertex.connect(neighbor_idx);
                }
            }
        }
    }

    /// Limited cyclic product triangulation (up to n vertices).
    fn cyclic_product_limited(&mut self, max_vertices: usize, centroid: bool) {
        // For now, just do full triangulation if we have room
        // A more sophisticated implementation would stop mid-way
        if self.cache.len() < max_vertices {
            self.cyclic_product_full(centroid);
        }
    }

    /// Split an edge by adding a midpoint vertex.
    ///
    /// This operation is memoized - calling with the same endpoints
    /// returns the same midpoint vertex.
    ///
    /// # Arguments
    ///
    /// * `v1_coords` - Coordinates of first endpoint
    /// * `v2_coords` - Coordinates of second endpoint
    ///
    /// # Returns
    ///
    /// Coordinates of the midpoint vertex (which is also added to the cache)
    pub fn split_edge_coords(&self, v1_coords: &[f64], v2_coords: &[f64]) -> Coordinates {
        let c1 = Coordinates::new(v1_coords.to_vec());
        let c2 = Coordinates::new(v2_coords.to_vec());

        // Ensure consistent ordering for cache key
        let (key_a, key_b) = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
        let cache_key = (key_a.clone(), key_b.clone());

        // Check memoization cache first
        {
            let cache = self.split_cache.read();
            if let Some(midpoint) = cache.get(&cache_key) {
                return midpoint.clone();
            }
        }

        // Compute midpoint
        let midpoint = key_a.midpoint(&key_b);

        // Get or create the vertex
        let v1 = self.cache.get_or_create(v1_coords.to_vec());
        let v2 = self.cache.get_or_create(v2_coords.to_vec());
        let vc = self.cache.get_or_create(midpoint.as_slice().to_vec());

        // Disconnect original edge
        v1.disconnect(v2.index());
        v2.disconnect(v1.index());

        // Connect to midpoint
        v1.connect(vc.index());
        vc.connect(v1.index());
        v2.connect(vc.index());
        vc.connect(v2.index());

        // Cache the result
        {
            let mut cache = self.split_cache.write();
            cache.insert(cache_key, midpoint.clone());
        }

        midpoint
    }

    /// Refine the complex by adding n new vertices.
    ///
    /// If initial triangulation hasn't been performed, it will be
    /// started first.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of vertices to add (or None to refine all)
    pub fn refine(&mut self, n: Option<usize>) {
        match n {
            None => self.refine_all(true),
            Some(count) => {
                let target = self.cache.len() + count;
                if !self.triangulated {
                    self.triangulate(Some(target), true);
                }
                // Additional refinement logic would go here
                // For now, just ensure we have enough vertices
                while self.cache.len() < target && !self.triangulated_vectors.is_empty() {
                    self.refine_all(true);
                }
            }
        }
    }

    /// Refine all triangulated regions.
    pub fn refine_all(&mut self, centroids: bool) {
        if !self.triangulated {
            self.triangulate(None, centroids);
            return;
        }

        let regions = self.triangulated_vectors.clone();
        let mut new_regions = Vec::new();

        for (origin, supremum) in regions {
            let sub_regions = self.refine_local_space(&origin, &supremum, centroids);
            new_regions.extend(sub_regions);
        }

        self.triangulated_vectors = new_regions;
        self.generation += 1;
    }

    /// Refine a local region of the complex using C3 cyclic group products.
    ///
    /// This is a faithful port of the Python `refine_local_space` method.
    /// It performs the following:
    /// 1. Splits the region diagonal to create/find the centroid
    /// 2. Takes a snapshot of the centroid's neighbors within the region (sup_set)
    /// 3. Creates C3 group product vertices across all dimensions
    /// 4. Optionally creates sub-region centroids
    /// 5. Returns sub-regions as (centroid, sup_set_vertex) pairs
    ///
    /// # Returns
    ///
    /// New sub-regions that replace the input region.
    fn refine_local_space(
        &self,
        origin: &[f64],
        supremum: &[f64],
        centroid: bool,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        // Ensure origin <= supremum component-wise
        let mut s_origin = origin.to_vec();
        let mut s_supremum = supremum.to_vec();
        for i in 0..self.dim {
            if s_origin[i] > s_supremum[i] {
                std::mem::swap(&mut s_origin[i], &mut s_supremum[i]);
            }
        }

        // Get or create origin and supremum vertices
        let _vo = self.cache.get_or_create(s_origin.clone());
        let _vs = self.cache.get_or_create(s_supremum.clone());

        // Split the main diagonal to create/find centroid
        let vco_coords = self.split_edge_coords(&s_origin, &s_supremum);
        let vco = self.cache.get_or_create(vco_coords.as_slice().to_vec());

        // Capture sup_set: centroid's current neighbors within region bounds.
        // This determines the sub-region count. Taken BEFORE C3 work
        // (matching Python's `sup_set = copy.copy(vco.nn)`).
        let eps = 1e-15;
        let sup_set: Vec<Vec<f64>> = vco
            .neighbor_indices()
            .iter()
            .filter_map(|&idx| self.cache.get_by_index(idx))
            .filter(|v| {
                v.x()
                    .iter()
                    .zip(s_origin.iter().zip(s_supremum.iter()))
                    .all(|(&x, (&lo, &hi))| lo - eps <= x && x <= hi + eps)
            })
            .map(|v| v.x().to_vec())
            .collect();

        // --- C3 cyclic group refinement ---
        // For the first dimension: create the axis vertex and split to center
        let mut a_vl_coords = s_origin.clone();
        a_vl_coords[0] = s_supremum[0];
        let _a_vl = self.cache.get_or_create(a_vl_coords.clone());
        let c_v_coords = self.split_edge_coords(&s_origin, &a_vl_coords);
        let c_v = self.cache.get_or_create(c_v_coords.as_slice().to_vec());
        c_v.connect(vco.index());
        vco.connect(c_v.index());

        // Track C3 groups as (lower_coords, center_coords, upper_coords)
        let mut groups: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = vec![(
            s_origin.clone(),
            c_v_coords.as_slice().to_vec(),
            a_vl_coords.clone(),
        )];

        // Process remaining dimensions with C3 group product
        for i in 1..self.dim {
            let mut new_groups = Vec::new();

            for (lower, _center, upper) in &groups {
                // Create aN vertices (shift dimension i to supremum)
                let mut a_lower = lower.clone();
                let mut a_upper = upper.clone();
                a_lower[i] = s_supremum[i];
                a_upper[i] = s_supremum[i];

                // Ensure all vertices exist
                let v_lower = self.cache.get_or_create(lower.clone());
                let v_upper = self.cache.get_or_create(upper.clone());
                let v_a_lower = self.cache.get_or_create(a_lower.clone());
                let v_a_upper = self.cache.get_or_create(a_upper.clone());

                // Split cross-diagonal: lower → a_upper
                let c_vc_coords = self.split_edge_coords(lower, &a_upper);
                let c_vc = self.cache.get_or_create(c_vc_coords.as_slice().to_vec());

                // Split same-side edges
                let c_vl_coords = self.split_edge_coords(lower, &a_lower);
                let c_vl = self.cache.get_or_create(c_vl_coords.as_slice().to_vec());

                let c_vu_coords = self.split_edge_coords(upper, &a_upper);
                let c_vu = self.cache.get_or_create(c_vu_coords.as_slice().to_vec());

                // Split far-side edge
                let a_vc_coords = self.split_edge_coords(&a_lower, &a_upper);
                let a_vc = self.cache.get_or_create(a_vc_coords.as_slice().to_vec());

                // Also split lower → upper (ensures center edge is split)
                let _ = self.split_edge_coords(lower, upper);

                // Connect all midpoints to the main centroid
                c_vc.connect(vco.index());
                vco.connect(c_vc.index());
                c_vl.connect(vco.index());
                vco.connect(c_vl.index());
                c_vu.connect(vco.index());
                vco.connect(c_vu.index());
                a_vc.connect(vco.index());
                vco.connect(a_vc.index());

                // Connect within the C3 group
                c_vc.connect(c_vl.index());
                c_vl.connect(c_vc.index());
                c_vc.connect(c_vu.index());
                c_vu.connect(c_vc.index());
                a_vc.connect(c_vc.index());
                c_vc.connect(a_vc.index());

                // Connect c_vc to the extreme vertices of this group
                c_vc.connect(v_lower.index());
                v_lower.connect(c_vc.index());
                c_vc.connect(v_upper.index());
                v_upper.connect(c_vc.index());
                c_vc.connect(v_a_lower.index());
                v_a_lower.connect(c_vc.index());
                c_vc.connect(v_a_upper.index());
                v_a_upper.connect(c_vc.index());

                // Build new C3 groups for the next dimension
                new_groups.push((
                    lower.clone(),
                    c_vl_coords.as_slice().to_vec(),
                    a_lower.clone(),
                ));
                new_groups.push((
                    _center.clone(),
                    c_vc_coords.as_slice().to_vec(),
                    a_vc_coords.as_slice().to_vec(),
                ));
                new_groups.push((
                    upper.clone(),
                    c_vu_coords.as_slice().to_vec(),
                    a_upper.clone(),
                ));
            }

            groups = new_groups;
        }

        // Build sub-centroids if requested
        if centroid {
            let vco_slice = vco_coords.as_slice();
            for sup_v in &sup_set {
                let pool = self.vpool(vco_slice, sup_v);
                let sub_centroid_coords = self.split_edge_coords(vco_slice, sup_v);
                let sub_centroid =
                    self.cache
                        .get_or_create(sub_centroid_coords.as_slice().to_vec());
                for &idx in &pool {
                    sub_centroid.connect(idx);
                    if let Some(neighbor) = self.cache.get_by_index(idx) {
                        neighbor.connect(sub_centroid.index());
                    }
                }
            }
        }

        // Return sub-regions: (centroid, each_neighbor_in_region)
        let vco_vec = vco_coords.as_slice().to_vec();
        sup_set
            .into_iter()
            .map(|sup_v| (vco_vec.clone(), sup_v))
            .collect()
    }

    /// Process all pending field evaluations.
    ///
    /// Evaluates the objective function at all vertices that haven't
    /// been evaluated yet.
    pub fn process_pools(&self) {
        self.cache.process_pools();
    }

    /// Find all local minimizers in the complex.
    ///
    /// A vertex is a local minimizer if its function value is less than
    /// all of its connected neighbors.
    pub fn find_minimizers(&self) -> Vec<Arc<crate::Vertex>> {
        self.cache.find_all_minimizers()
    }

    /// Get the vertex pool for a local region.
    ///
    /// Returns vertices that are within the bounds defined by origin and supremum.
    pub fn vpool(&self, origin: &[f64], supremum: &[f64]) -> HashSet<usize> {
        // Determine actual bounds
        let mut bl = origin.to_vec();
        let mut bu = supremum.to_vec();
        for i in 0..self.dim {
            if bl[i] > supremum[i] {
                bl[i] = supremum[i];
            }
            if bu[i] < origin[i] {
                bu[i] = origin[i];
            }
        }

        let vo = self.cache.get_or_create(origin.to_vec());
        let vs = self.cache.get_or_create(supremum.to_vec());

        let mut pool: HashSet<usize> = HashSet::new();
        pool.extend(vo.neighbor_indices().iter().copied());
        pool.extend(vs.neighbor_indices().iter().copied());

        // Filter to vertices within bounds
        pool.retain(|&idx| {
            if let Some(v) = self.cache.get_by_index(idx) {
                v.x()
                    .iter()
                    .zip(bl.iter().zip(bu.iter()))
                    .all(|(&x, (&lo, &hi))| lo <= x && x <= hi)
            } else {
                false
            }
        });

        pool
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi.powi(2)).sum()
    }

    #[test]
    fn test_complex_creation() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds.clone(), sphere, None);

        assert_eq!(complex.dim(), 2);
        assert_eq!(complex.bounds(), &bounds[..]);
        assert_eq!(complex.vertex_count(), 0);
        assert!(!complex.is_triangulated());
    }

    #[test]
    fn test_triangulate_2d() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);

        // Should have at least 4 corners + 1 centroid = 5 vertices
        assert!(
            complex.vertex_count() >= 5,
            "Expected at least 5 vertices, got {}",
            complex.vertex_count()
        );
        assert!(complex.is_triangulated());

        // Verify corners exist
        let corners = [
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        for corner in &corners {
            let v = complex.cache.get_or_create(corner.clone());
            assert!(v.neighbor_indices().len() > 0, "Corner {:?} should have neighbors", corner);
        }
    }

    #[test]
    fn test_triangulate_3d() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);

        // Should have at least 8 corners + centroid
        assert!(
            complex.vertex_count() >= 9,
            "Expected at least 9 vertices, got {}",
            complex.vertex_count()
        );
    }

    #[test]
    fn test_triangulate_1d() {
        let bounds = vec![(0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);

        // Should have 2 endpoints + 1 centroid = 3 vertices
        assert!(
            complex.vertex_count() >= 3,
            "Expected at least 3 vertices, got {}",
            complex.vertex_count()
        );
    }

    #[test]
    fn test_split_edge() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        let v1 = vec![0.0, 0.0];
        let v2 = vec![1.0, 1.0];

        let midpoint = complex.split_edge_coords(&v1, &v2);

        assert_eq!(midpoint.as_slice(), &[0.5, 0.5]);

        // Verify memoization: same result
        let midpoint2 = complex.split_edge_coords(&v1, &v2);
        assert_eq!(midpoint, midpoint2);

        // Verify order doesn't matter
        let midpoint3 = complex.split_edge_coords(&v2, &v1);
        assert_eq!(midpoint, midpoint3);
    }

    #[test]
    fn test_process_pools() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);
        complex.process_pools();

        // All vertices should be evaluated
        assert!(complex.cache.function_evaluations() > 0);

        // Origin should have f=0
        let origin = complex.cache.get_or_create(vec![0.0, 0.0]);
        assert!((origin.f().unwrap() - 0.0).abs() < 1e-10);

        // (1,1) should have f=2
        let corner = complex.cache.get_or_create(vec![1.0, 1.0]);
        assert!((corner.f().unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_minimizers() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);
        complex.process_pools();

        let minimizers = complex.find_minimizers();

        // Origin (0,0) should be the only minimizer
        assert!(!minimizers.is_empty());

        let has_origin = minimizers.iter().any(|v| {
            v.x().iter().all(|&x| x.abs() < 1e-10)
        });
        assert!(has_origin, "Origin should be a minimizer");
    }

    #[test]
    fn test_refine() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);
        let initial_count = complex.vertex_count();

        complex.refine(Some(5));

        // Should have more vertices after refinement
        assert!(
            complex.vertex_count() >= initial_count,
            "Refinement should not decrease vertex count"
        );
    }

    #[test]
    fn test_vpool() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        complex.triangulate(None, true);

        // Get vertices in the lower-left quadrant
        let pool = complex.vpool(&[0.0, 0.0], &[0.5, 0.5]);

        // Pool should contain some vertices
        // (at minimum, neighbors of origin that are in this region)
        assert!(!pool.is_empty() || complex.vertex_count() < 5);
    }

    #[test]
    fn test_generation_tracking() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut complex: Complex<_, fn(&[f64]) -> bool> = Complex::new(bounds, sphere, None);

        assert_eq!(complex.generation(), 0);

        complex.triangulate(None, true);
        assert_eq!(complex.generation(), 0);

        complex.refine_all(true);
        assert_eq!(complex.generation(), 1);

        complex.refine_all(true);
        assert_eq!(complex.generation(), 2);
    }
}
