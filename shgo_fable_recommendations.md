# SHGO-RS Native: Analysis & Recommendations

*Prepared 2026-07-22 (Claude Fable 5). Fidelity claims below were checked against the
current SciPy `main` source of `scipy/optimize/_shgo.py`, not from memory.*

*Revised same day after fetching origin: the initial version of this document was written
against a stale local checkout (2ec07b4, Feb 2026) and wrongly concluded the KNN work
didn't exist. GitHub master was three commits ahead (157e8fd, April 2026), containing the
connectivity work — see §1 and §6. Line references and findings below have been
re-verified against 157e8fd; items the April commits already fixed are marked.*

## Executive summary

The implementation is structurally faithful to SciPy SHGO and in good health: 90 unit
tests and 10 cross-validation tests pass, the Sobol generator is bit-exact against
`scipy.stats.qmc.Sobol`, and the LCB construction, stopping-criteria reconciliation, and
the `vf_to_vv` connectivity quirk are all exact ports. The rayon parallel local-minimization
pool is a well-executed extension.

However:

1. There is a **single dominant scalability bug** — O(V) vertex-index lookups making
   minimizer detection O(V²·degree) — still present at 157e8fd despite the April
   "performance optimizations" commit. It has a trivial O(1) fix (§3.1).
2. `cargo test` actually **fails**: all 21 doctests are broken (import name mismatch).
   The "100 tests pass" from the v1 commit counted only unit + integration tests.
3. The C FFI **silently ignores constraints**, and a handful of options are dead.
4. Several fidelity deviations from SciPy exist, mostly affecting multi-iteration runs
   and evaluation accounting; notably, connectivity graphs (all four methods) are built
   per-batch rather than cumulatively (§4.2).

---

## 0. Fix status (2026-07-22 working session)

All four §3 critical issues were fixed and validated the same day, with a
before/after harness (`examples/fix_benchmark.rs`, JSON output; baselines in the
session scratchpad). Accuracy fields (`fun`, `x`, `funl`, minima count, `nit`) were
**bitwise identical** before and after across all six cases; `nfev` changed only by the
intended `+ nlfev` (§3.4). Timing (min of 2 reps, 22-core machine):

| Case | Before | After | Speedup |
|---|---|---|---|
| Simplicial sphere 9-D (V ≈ 20k) | 167.2 s | 0.51 s | **~326×** |
| Sobol+KNN sphere 6-D, n=16384, 2 iters | 4.36 s | 2.55 s | 1.7× (remainder = serial KNN build, P1) |
| Sobol+KNN rastrigin 6-D, n=4096 | 0.22 s | 0.19 s | — |
| Sobol Delaunay / 10-D KNN / constrained | ≤ 0.13 s | ≤ 0.12 s | — |

The 167 s → 0.5 s case was dominated by `Complex::vpool`'s serial `get_by_index`
linear scans during refinement (a hub vertex with ~26k neighbors, scanned O(V) per
neighbor per sub-region) — which also explains why pre-fix runs pegged a single core.
Test suite after all fixes: 100 unit + 10 cross-validation + 21 doctests, all passing.

**P1 follow-up (same session, committed separately):** the brute-force KNN build is now
rayon-parallel across rows (edge set unchanged — deterministic); the three ANN
connectivity methods (KNN/HNSW/ScaNN) build over the **full cumulative point cloud**
each iteration, matching SciPy's re-triangulate-everything semantics (Delaunay remains
per-batch — the P2 item); and `maxev` now counts sampled points (SciPy's `n_sampled`),
not feasible evaluations, with a discriminating unit test
(`test_maxev_counts_sampled_points_not_fev`). Validation: Delaunay and simplicial paths
stayed bitwise identical; KNN multi-iteration results changed in the intended direction
(16384-point sphere drops a duplicate near-origin candidate, 2 → 1 minima and fewer
wasted local evals; rastrigin cases surface additional distinct minima). Timing:
Sobol+KNN sphere 6-D n=16384 ×2 iters: 2.55 s → **1.22 s** (3.6× vs original baseline).
Suite: 101 unit + 10 cross-validation + 21 doctests green.

## 1. The KNN / connectivity work: exists on GitHub since April 2026

Commit `157e8fd` ("Add 4 connectivity methods for Sobol mode + performance
optimizations", 2026-04-05, current GitHub master) adds a `ConnectivityMethod` enum —
`Delaunay` (default) | `KNearestNeighbors` | `HNSW` | `ScaNN` — plus
`ShgoOptions.knn_neighbors: Option<usize>` (default k = 2·dim+1). Two companion commits
(2026-04-03) add a Delaunay joggle retry for degenerate point sets and rework local
optimizer selection (`ShgoOptions.local_optimizer` was **removed**; the algorithm now
lives in `local_options.algorithm`, with auto-upgrade to COBYLA when constraints are
present). This local checkout was three commits behind and has been fast-forwarded.

Downstream: the tcdts algotrader (`~/src/tcdts/trader`) pins exactly `157e8fd` in its
Cargo.lock via the git dependency, so **the trader build has the KNN capability**, and
its config→enum mapping ("knn"/"knearestneighbors", "hnsw", "scann", default delaunay)
matches the crate. Beware the *second*, divergent checkout at `~/src/tcdts/shgo-rs-native`:
it carries two unpushed experimental commits (VertexCache pruning + revert, d5e6dce/d935125)
branched from before the connectivity commit, and is not used by the trader build.

§6 assesses the quality of the connectivity implementations and what remains to improve.

## 2. What is verified and solid

Credit where due — these were checked line-by-line against SciPy:

- **Sobol sequences**: Joe-Kuo direction numbers via `sobol-qmc` match SciPy exactly;
  fixtures verify 2D/3D/5D values to 1e-10. Power-of-2 rounding of `n` matches SciPy.
  Sequence continuation across iterations (`skip = sobol_skip + total_points`) is correct.
- **`construct_lcb_simplicial`** (src/shgo.rs:1202): exact port of SciPy's neighbor-based
  bound tightening. **`construct_lcb_delaunay`** equivalent (global bounds in Sobol mode)
  also matches.
- **Stopping criteria**: `effective_iters` (iters disabled when any other criterion is
  set) matches SciPy's `__init__` logic; the `f_min`/`f_tol` formula matches
  `finite_precision` exactly, including the `f_min == 0` special case. Current SciPy uses
  `iters_done >= iters` (no off-by-one), and the Rust `iteration >= iters` agrees.
- **`vf_to_vv` quirk**: SciPy connects only `e[0]–e[1]` of each dim-combination per
  simplex (so for dim ≥ 3 only edges among the first three vertices of each simplex are
  added). The Rust port reproduces this faithfully rather than "fixing" it — the right
  call for a fidelity port.
- **Thread safety**: the coordinate-keyed cache with double-checked locking, per-start-point
  `LMapCache` memoization, and fresh NLopt instances per call (NLopt is `!Send`) are all
  sound. FFI memory management (alloc/free pairing, partial-failure cleanup) is careful.

---

## 3. Critical issues

### 3.1 O(V²) minimizer detection — the scalability killer — **FIXED (2026-07-22)**

`VertexCache::get_by_index` is a linear scan over the whole map:

```rust
// src/vertex.rs:468
pub fn get_by_index(&self, index: usize) -> Option<Arc<Vertex>> {
    self.cache.read().values().find(|v| v.index() == index).cloned()
}
```

The same `values().find(...)` pattern is inlined in the hot paths (line numbers at 157e8fd):

- `is_minimizer` / `is_maximizer` — per *neighbor* lookup (src/vertex.rs:601, 633)
- `process_field` / `process_constraints` — per pending vertex (src/vertex.rs:500, 531)
- `construct_lcb_simplicial` → `get_by_index` per neighbor (src/shgo.rs:1593)

The April perf commit (SmallVec coordinates, lock-free AtomicU8 extrema caching) did
**not** touch this; the quadratic scan is still the asymptotic bottleneck. At the
trader's `n = 4096, k = 40, maxiter = 2` this is ~10⁹ map-entry visits in iteration 2
alone; at `n = 32768` it reaches ~10¹¹ — pure overhead that grows quadratically with n.

`find_all_minimizers` is therefore O(V² · degree). Concretely: a 6-D Sobol run with
n = 2048 does ~2048 × degree × 2048 ≈ 10⁸ pointer-chasing comparisons *per iteration*; a
dim-10 simplicial refinement (~59k vertices) reaches ~10¹³. This alone explains "larger
parameter optimizations became computationally infeasible."

**Fix (trivial):** vertex `index` equals insertion position — it is assigned from
`next_index.fetch_add` *inside the same write-lock section* that inserts into the
`IndexMap`, and vertices are never removed. So:

```rust
pub fn get_by_index(&self, index: usize) -> Option<Arc<Vertex>> {
    self.cache.read().get_index(index).map(|(_, v)| v.clone())
}
```

and replace every inline `values().find(|v| v.index() == idx)` with the same call.
This turns minimizer scans O(V·degree) — likely a 100–10,000× speedup at the sizes you
care about. Add a debug assertion (`debug_assert_eq!(vertex.index(), map_position)`) to
guard the invariant.

### 3.2 All 21 doctests fail — `cargo test` is red — **FIXED (2026-07-22)**

Every doc example imports `use shgo_rs::…`, but the library target is named `shgo`
(`Cargo.toml [lib] name = "shgo"`), so every doctest dies with E0432
(e.g. src/lib.rs:16, src/coordinates.rs:212). Unit and integration tests pass
(90 + 10), so this went unnoticed — the v1 commit's "All 100 tests pass" never counted
doctests. Fix: `sed -i 's/use shgo_rs::/use shgo::/'` across `src/`, then keep
`cargo test` (all targets) green. Consider a CI workflow so this can't regress.

### 3.3 C FFI silently drops constraints — **FIXED (2026-07-22)**

`shgo_add_constraint` stores constraints in the handle, but `shgo_minimize` never uses
them:

```rust
// src/ffi.rs:396–398
// Run optimization - constraints not directly supported in this API version
// For now we run without constraints (constraints need special handling in SHGO)
let shgo = Shgo::new(objective_wrapper, handle.bounds.clone())...
```

A C/C++ user who registers constraints gets an *unconstrained* answer with
`SHGO_SUCCESS`. Either wire the stored constraints through `Shgo::with_constraints`
(the Rust side fully supports them), or make `shgo_add_constraint` return an
"unsupported" status until then. Silent wrong answers are the worst failure mode here.

### 3.4 `nfev` under-reports: local-minimization evaluations are dropped — **FIXED (2026-07-22)**

Both iterate paths end with `result.nfev += lmap_cache.total_fev()`
(src/shgo.rs:868, 1127) — but `minimize_inner` then unconditionally overwrites it:

```rust
// src/shgo.rs:635
result.nfev = self.fev_count.load(Ordering::Relaxed);
```

making those `+=` lines dead code. SciPy reports `res.nfev = self.fn + self.res.nlfev`
(sampling + local evaluations). Fix: after line 541, add `result.nfev += result.nlfev`
(or `lmap_cache.total_fev()` — note `nlfev` currently only sums *successful* minimizations
while `total_fev` counts all; SciPy's LMC counts all, so prefer `total_fev`, which means
returning it from the iterate functions or hoisting the cache).

---

## 4. Fidelity deviations from SciPy

| # | Deviation | Where | Impact |
|---|---|---|---|
| 4.1 | ~~`maxev` compares against function evaluations instead of sampled points~~ **FIXED (2026-07-22)** | — | SciPy checks `n_sampled >= maxev` (sampling points incl. infeasible); Rust reuses `fev_count`, making `maxev` a duplicate of `maxfev`. Fix: track points generated / vertex-cache size. |
| 4.2 | Sobol mode builds connectivity over only the new batch — **fixed for KNN/HNSW/ScaNN (2026-07-22); Delaunay still per-batch (P2)** | src/shgo.rs:995–1060 | SciPy accumulates all points in `self.C` and re/incrementally triangulates the whole cloud (`Tri.add_points`), so old vertices gain new neighbors and stale minimizers get disqualified. Rust leaves old vertices with stale neighborhoods and never connects old↔new; the new KNN/HNSW/ScaNN builders inherit the same per-batch scope. Single-iteration runs are unaffected; multi-iteration runs produce extra local-minimization candidates (LMC dedup keeps this cheap, but it deviates from SciPy). For KNN this is trivially fixable: build the graph over *all* cached vertices each iteration. |
| 4.3 | Default simplicial iteration refines the whole complex | src/shgo.rs:766–772, src/complex.rs:410–444 | With default `n=0`, every iteration calls `refine_all` → super-exponential vertex growth (≈3^dim per generation). SciPy's default resolves `n = 2^dim + 1` and calls `refine(n)`, adding ~n targeted points per iteration (linear growth). Combined with 3.1 this is the other half of the scaling wall. |
| 4.4 | `Complex::refine(Some(n))` overshoots; `cyclic_product_limited` ignores its limit | src/complex.rs:337–344, 410–425 | `refine(n)` loops `refine_all` until `len ≥ target` (can add thousands for n=5); the initial triangulation always builds all 2^dim corners regardless of `n` (1M vertices at dim=20 before refinement starts). SciPy's `Complex.triangulate(n)`/`refine(n)` are incremental and bounded. This makes simplicial mode unusable beyond dim ≈ 15 even for tiny n. |
| 4.5 | One extra refinement generation in every simplicial run | src/shgo.rs:754–772 | Rust triangulates *before* the loop and refines *inside* iteration 1; SciPy's first iteration is the initial triangulation only. Default `iters=1` therefore samples ~3^dim points where SciPy samples 2^dim+1 — different nfev, different minimizer pool. The cross-validation suite doesn't catch it because there is no end-to-end fixture (see §7). |
| 4.6 | `maxiter_local` truncates an unsorted pool | src/shgo.rs:802, 1072 | SciPy sorts the minimizer pool by function value (`sort_min_pool`) before trimming; Rust `.take(maxiter_local)` grabs insertion-order candidates. Fix: sort candidates by vertex f before `take`. (Abandoning `g_topograph` distance ordering for the parallel pool is fine — order no longer affects results because all candidates run — but *which* candidates survive truncation should match.) |
| 4.7 | Unconverged local results counted as successes | src/local_opt.rs:341, 501 | NLopt `MaxevalReached`/`MaxtimeReached` return `Ok` → `success: true`, so unconverged points enter `xl`. SciPy filters on `lres.success`. Consider mapping maxeval/maxtime termination to `success=false` (or a separate flag) to match. |
| 4.8 | Dead options: `symmetry`, `min_feasible_ratio`, `infty_constraints` | src/shgo.rs:200–278 | Declared, documented in README, never read. Worse, `symmetry` defaults to `true` while SciPy defaults symmetry *off* — a user reading the README expects an active feature. `infty_constraints` is hardcoded as `1e50` in `process_bounds`. Either implement or delete the fields; deleting is honest. |
| 4.9 | `minhgrd` (homology-growth stopping) not ported | — | Niche SciPy option; document as unsupported. |
| 4.10 | `xl` dedup uses absolute 1e-10 tolerance | src/shgo.rs:640–650 | An extension (SciPy keeps per-start entries). Fine, but absolute tolerance is scale-dependent: distinct minima closer than 1e-10 merge; at bounds like ±1e50, nothing ever merges meaningfully. Use a relative tolerance or document. |
| 4.11 | ~~Delaunay failure aborts the whole run~~ **FIXED in 32f5edb** (April 2026) | — | A joggle retry now handles cocircular/degenerate point sets, and the KNN/HNSW/ScaNN methods avoid qhull entirely. |
| 4.12 | Budget checks only between iterations | src/shgo.rs:861, 1120 | SciPy can break inside the minimizer-pool loop; the parallel pool can't, so `maxfev`/`maxtime` overshoot by up to a full pool. Acceptable consequence of parallelism — document it, or cap pool size when a budget is near. |
| 4.13 | ~~`local_options.algorithm` silently overwritten by `local_optimizer`~~ **FIXED in 70ec80d** (April 2026) | — | The redundant `ShgoOptions.local_optimizer` field was removed; `local_options.algorithm` is now authoritative, with auto-upgrade to COBYLA (plus a `disp` warning) when constraints require it. Note this is an **API break** for consumers of the old field. |

## 5. Smaller defects and hygiene

- **`unsafe impl Send/Sync for Vertex`** (src/vertex.rs:315–316): unnecessary — every
  field is already Send+Sync (`RwLock`, `AtomicBool`, plain data). Delete both lines and
  let the compiler verify; keeping them suppresses real errors if a non-Sync field is
  ever added.
- **FFI options mismatches** (src/ffi.rs:94–150): `ShgoOptions_C.f_tol` is documented as
  "function tolerance for local minimization" with default 1e-12, but it maps onto
  `ShgoOptions.f_tol` — the *f_min stopping* tolerance (Rust default 1e-4). Local optimizer
  tolerances aren't settable via FFI at all. Also missing: `iters`, `maxev`, `f_min`,
  `sobol_skip`. And `n` defaults to 128 in C even for simplicial mode, where the Rust
  default (0 = auto 2^dim+1) differs — C and Rust defaults silently diverge.
- **README inaccuracies**: the `OptimizeResult` table says `xl: Vec<LocalMinimum>` and the
  "Accessing All Local Minima" example calls `lm.fun` on elements of `result.xl` — but
  `xl` is `Vec<Vec<f64>>`, so the example doesn't compile. The verification section
  overstates coverage (see §7). `shgo-rs = "0.1"` is not actually published to crates.io.
- **Dead condition** `effective_n == 0` in src/shgo.rs:768 (`effective_n()` never returns 0).
- **`validate_bounds` NaN check is dead** (src/shgo.rs:700): NaNs were already replaced by
  `process_bounds` in the constructor.
- **SmallVec optimization caps at dim 8** (src/coordinates.rs): `SmallVec<[f64; 8]>`
  spills to the heap for dim > 8, which is precisely the "larger parameter" regime this
  crate targets — consider `[f64; 16]` or making it a feature choice (measure first;
  larger inline arrays also grow every map entry).
- **Unused `parallel` feature flag** in Cargo.toml (rayon is unconditional).
- **Clippy**: ~30 warnings (unused imports, never-read `ShgoHandle.bounds`-style fields,
  deprecated `criterion::black_box`, complex types). `cargo clippy --fix` handles most.
- **AppleDouble junk**: untracked `._*` files throughout the worktree (macOS copy
  artifacts). Add `._*` and `.DS_Store` to `.gitignore` and delete them.
- **`LMapCache::get_sorted`** treats NaN comparisons as Equal — benign today, but a NaN
  objective at a "successful" local minimum would sort arbitrarily and could be selected
  by the empty-`xl` fallback.

## 6. Assessment of the connectivity backends (157e8fd)

The April implementation matches what §1's design would have called for. Per-method
review (all in src/shgo.rs:1230–1462):

**KNearestNeighbors** (`build_knn_connectivity`) — the right default for this crate's
problem sizes. Correct: bidirectional (symmetrized) connections, self-connection guarded
by `Vertex::connect`, k capped at n−1, duplicate points collapse to the same cache
vertex harmlessly, `select_nth_unstable` partial sort is the right O(n) per-row choice.
Two improvements available:

1. The outer loop is **serial**. O(n²·d) at n = 32768, d ≈ 15 is ~1.5×10¹⁰ ops ≈
   tens of seconds per iteration, single-threaded. The rows are independent — a
   `par_iter` over `i` (collecting edge lists, then connecting) is a free ~10× win.
2. Per-batch scope (deviation 4.2): building over all cached vertices instead of the
   current batch would both match SciPy semantics more closely and disqualify stale
   candidates.

**HNSW** (`build_hnsw_connectivity`, via `hnsw_rs`) — reasonable parameters
(ef_search ≥ k, ef_construction = max(4k, 48)). Notes: results are approximate (recall
< 100% → occasional missed/extra minimizer candidates — benign, LMC-deduped); each
query returns the point itself among the k, so effective neighbor count is k−1
(inconsistent with brute-force KNN's k true neighbors); the code skips
`set_searching_mode` after `parallel_insert` with a comment acknowledging the wrinkle —
worth verifying against `hnsw_rs` docs that concurrent search post-insert is defined
behavior. Only pays off vs brute-force KNN at n well above ~10⁵ — at trader scales
(4096–32768), plain KNN is both exact and fast enough.

**ScaNN** (`build_scann_connectivity`, via `vecstore`) — accuracy-first configuration
(`num_leaves_to_search = num_leaves`, i.e., searches *all* leaves) means it has **no
speed advantage** over brute-force KNN here; it's effectively quantized (f32,
8-bit codes) exhaustive search with rerank. Graceful degradation is good (falls back to
KNN on index/train/add failure), but per-point query errors are silently skipped —
a vertex with zero neighbors vacuously passes the minimizer test (`all()` over an empty
set) and spawns a spurious local minimization. Recommendation: treat ScaNN as
experimental; prefer KNN.

**Delaunay** — joggle retry added (4.11 fixed); still the SciPy-faithful default and the
right choice below dim ~7.

**Recommended configuration** for large parameter counts: `KNearestNeighbors` with
k ≈ 2·dim+1 up to ~3·dim; larger k (e.g. 40 at dim ≈ 15) densifies the graph, which
*reduces* spurious minimizer candidates (fewer false basins) at the cost of possibly
merging genuinely distinct shallow basins — a sensible trade for expensive objectives.

**Remaining bottleneck ordering** once KNN is in use: (1) the O(V²) index lookups of
§3.1 — now the single dominant cost at large n; (2) serial KNN construction; (3) the
per-batch graph scope. With all three addressed, per-iteration cost is
O(V·k) minimizer detection + O(V²·d / cores) graph build + parallel local minimizations,
which comfortably supports dim 10–50 at n = 10⁴–10⁵.

## 7. Verification gaps

The README claims final results are "verified to match SciPy output bit-for-bit" — the
fixture suite doesn't support that yet. What `tests/fixtures/` actually covers: Sobol
values (2/3/5-D, genuinely exact), midpoint arithmetic, sphere evaluations at fixed
points, one 4-neighbor star-pattern minimizer check, and initial-triangulation vertex
counts (2-D = 5). There is **no end-to-end `shgo()` fixture** (final x/fun/xl/nfev per
benchmark function), no Delaunay-simplices fixture, and no refinement-structure fixture —
which is exactly why deviations 4.2–4.5 went undetected.

Recommended additions, in order of value:

1. End-to-end fixtures: for sphere/rosenbrock/rastrigin/eggholder × {simplicial, sobol}
   × {iters=1, maxiter=3}, record SciPy's `x`, `fun`, sorted `funl`, `nit`, `nfev`;
   assert Rust matches (values to tolerance; counts exactly once 4.5 is resolved).
2. A refinement fixture: vertex coordinates + adjacency after one `refine_all` on the
   unit square/cube, compared structurally.
3. Fix `tests/generate_fixtures.py` setup: it expects a stefan-endres/shgo checkout at
   `../../shgo` (that path is currently an unrelated Rust stub) — prefer
   `pip install shgo` + plain `import shgo`, and document the requirement.

## 8. Prioritized action plan

| Priority | Item | Effort |
|---|---|---|
| ~~P0~~ ✅ | 3.1 O(1) `get_by_index` + remove inline linear scans | done |
| ~~P0~~ ✅ | 3.2 doctest imports (`shgo_rs` → `shgo`), full `cargo test` green | done |
| ~~P0~~ ✅ | 3.3 FFI: constraints wired through + regression test | done |
| ~~P1~~ ✅ | 6. parallelize the brute-force KNN loop (rayon over rows) | done |
| ~~P1~~ ✅ | 6/4.2 build ANN graphs over all cached vertices (cumulative) | done |
| ~~P1~~ ✅ | 3.4 nfev accounting; 4.1 maxev semantics | done |
| P1 | 4.3/4.4/4.5 simplicial growth control (`refine(n)` bounded, no extra generation) | ~1 day |
| P2 | 4.2 cumulative/incremental Delaunay per iteration | ~0.5 day |
| P2 | 4.6 sort pool before `maxiter_local`; 4.7 success semantics; 4.8 delete dead options | ~2 h |
| P2 | 7. end-to-end SciPy fixtures + fixture-generation setup | ~0.5 day |
| P2 | 6. ScaNN: fail loudly on per-point query errors (isolated-vertex → spurious candidate) | ~30 min |
| P3 | §5 hygiene: unsafe impls, clippy, README corrections, `.gitignore` `._*`, FFI option docs, push-or-discard the divergent tcdts checkout commits | ~2 h |

The KNN capability itself shipped in April (157e8fd) and the trader is pinned to it.
The P0 row — above all the O(1) lookup fix — is now the shortest path to making the
larger runs (n = 16384–32768) fast, since minimizer detection is the remaining
quadratic term.
