//! Local-optimizer shoot-out: BOBYQA vs L-BFGS vs Nelder-Mead vs Subplex vs CMA-ES.
//!
//! The landscapes are chosen so that each one breaks a specific assumption of
//! the model-based local methods, which is where a population method earns
//! its extra evaluations:
//!
//! | problem  | what it stresses                                       | expected winner |
//! |----------|--------------------------------------------------------|-----------------|
//! | rosen    | smooth curved valley (control)                         | BOBYQA / L-BFGS |
//! | rotated  | rotated ellipsoid, condition 1e6 (non-separable)       | BOBYQA, CMA-ES  |
//! | rugged   | bowl + small high-frequency ripples (~1e4 local minima)| CMA-ES (4x pop) |
//! | noisy    | ellipsoid + frozen white noise, 10x the target scale   | CMA-ES (4x pop) |
//! | step     | quantised ellipsoid (flat plateaus, zero gradient)     | Subplex, CMA-ES |
//!
//! CMA-ES is run twice: with the crate's default population
//! (`4 + floor(3 ln dim)`) and with four times that, the standard remedy for
//! multimodal or noisy landscapes (IPOP-style).
//!
//! Part A runs every optimizer alone from the same fixed starting points.
//! Part B runs the same problems through SHGO (Sobol + k-NN, coarse `n`) so
//! the interaction with the minimizer pool is visible. Both parts print
//! Markdown tables.
//!
//! Run: cargo run --release --example local_optimizer_benchmark [quick]

use shgo::local_opt::{minimize_local, CmaesOptions, LocalOptimizer, LocalOptimizerOptions};
use shgo::{ConnectivityMethod, SamplingMethod, Shgo, ShgoOptions};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Landscapes
// ---------------------------------------------------------------------------

/// Off-lattice optimum so no coordinate sits on a Sobol point or a plateau edge.
fn shift(dim: usize) -> Vec<f64> {
    (0..dim).map(|i| 0.7 * ((i + 1) as f64 * 1.3).sin()).collect()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

/// Householder reflection `I - 2 v vᵀ / (vᵀ v)` with a dense fixed `v`: an
/// orthogonal matrix that couples every coordinate.
fn householder(dim: usize) -> Vec<Vec<f64>> {
    let v: Vec<f64> = (0..dim).map(|i| 1.0 + ((i + 1) as f64 * 0.9).cos()).collect();
    let vv: f64 = v.iter().map(|a| a * a).sum();
    (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let d = if i == j { 1.0 } else { 0.0 };
                    d - 2.0 * v[i] * v[j] / vv
                })
                .collect()
        })
        .collect()
}

fn ellipsoid_weights(dim: usize, log10_cond: f64) -> Vec<f64> {
    (0..dim)
        .map(|i| 10f64.powf(log10_cond * i as f64 / (dim - 1).max(1) as f64))
        .collect()
}

fn rotated_ellipsoid(x: &[f64], c: &[f64], r: &[Vec<f64>], w: &[f64]) -> f64 {
    let d: Vec<f64> = x.iter().zip(c).map(|(a, b)| a - b).collect();
    r.iter()
        .zip(w)
        .map(|(row, wi)| {
            let y: f64 = row.iter().zip(&d).map(|(a, b)| a * b).sum();
            wi * y * y
        })
        .sum()
}

/// Sphere plus small ripples of period `1/k` and amplitude `2a`. Local minima
/// exist wherever the ripple slope beats the bowl slope, i.e. for
/// `|x_i - c_i| < pi k a` — with `a = 0.15, k = 3` that is a radius of 1.4
/// around the optimum, about `8^dim` local minima. The nearest non-global
/// minimum (one ripple off in one coordinate) has `f ~ 0.11`. Global minimum
/// 0 at `c`.
const RUGGED_A: f64 = 0.15;
const RUGGED_K: f64 = 3.0;
fn rugged_bowl(x: &[f64], c: &[f64]) -> f64 {
    x.iter()
        .zip(c)
        .map(|(a, b)| {
            let d = a - b;
            d * d + RUGGED_A * (1.0 - (2.0 * std::f64::consts::PI * RUGGED_K * d).cos())
        })
        .sum()
}

/// Deterministic "frozen" white noise in [-1, 1] from the bits of `x`.
fn frozen_noise(x: &[f64]) -> f64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for &xi in x {
        h ^= xi.to_bits();
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 31;
    }
    (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Additive noise of amplitude 0.5: fifty times the objective's value (0.01)
/// at the success radius 0.1 on the weakest axis, so reaching that radius
/// requires averaging rather than trusting individual evaluations.
const NOISE_AMPLITUDE: f64 = 0.5;
fn noisy_ellipsoid(x: &[f64], c: &[f64], w: &[f64]) -> f64 {
    let g: f64 = x
        .iter()
        .zip(c)
        .zip(w)
        .map(|((a, b), wi)| wi * (a - b).powi(2))
        .sum();
    g + NOISE_AMPLITUDE * frozen_noise(x)
}

/// Ellipsoid on coordinates quantised to plateaus of width 0.1 (BBOB f7
/// style). Zero on the plateau containing `c`; zero gradient everywhere.
fn step_ellipsoid(x: &[f64], c: &[f64], w: &[f64]) -> f64 {
    x.iter()
        .zip(c)
        .zip(w)
        .map(|((a, b), wi)| {
            let q = ((a - b) * 10.0).round() / 10.0;
            wi * q * q
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Problem set
// ---------------------------------------------------------------------------

type Objective = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

#[derive(Clone, Copy, PartialEq)]
enum Metric {
    /// `f(x) - f*`
    FunGap,
    /// `||x - x*||` (for noisy objectives, where `f` itself is unreliable)
    Distance,
}

struct Problem {
    name: &'static str,
    dim: usize,
    bounds: Vec<(f64, f64)>,
    f: Objective,
    x_star: Vec<f64>,
    f_star: f64,
    metric: Metric,
    /// error below which a run counts as having reached the optimum
    success_tol: f64,
    /// SHGO sampling size for part B
    shgo_n: usize,
}

impl Problem {
    fn error(&self, x: &[f64], fun: f64) -> f64 {
        match self.metric {
            Metric::FunGap => fun - self.f_star,
            Metric::Distance => x
                .iter()
                .zip(&self.x_star)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),
        }
    }
}

fn problems(quick: bool) -> Vec<Problem> {
    let mut v = Vec::new();

    {
        let dim = 6;
        v.push(Problem {
            name: "rosen-6d",
            dim,
            bounds: vec![(-2.0, 2.0); dim],
            f: Arc::new(rosenbrock),
            x_star: vec![1.0; dim],
            f_star: 0.0,
            metric: Metric::FunGap,
            success_tol: 1e-6,
            shgo_n: 256,
        });
    }
    {
        let dim = 10;
        let c = shift(dim);
        let r = householder(dim);
        let w = ellipsoid_weights(dim, 6.0);
        let c2 = c.clone();
        v.push(Problem {
            name: "rotated-10d",
            dim,
            bounds: vec![(-5.0, 5.0); dim],
            f: Arc::new(move |x: &[f64]| rotated_ellipsoid(x, &c2, &r, &w)),
            x_star: c,
            f_star: 0.0,
            metric: Metric::FunGap,
            success_tol: 1e-6,
            shgo_n: 512,
        });
    }
    for &dim in if quick { &[4usize][..] } else { &[4usize, 10][..] } {
        let c = shift(dim);
        let c2 = c.clone();
        v.push(Problem {
            name: if dim == 4 { "rugged-4d" } else { "rugged-10d" },
            dim,
            bounds: vec![(-5.0, 5.0); dim],
            f: Arc::new(move |x: &[f64]| rugged_bowl(x, &c2)),
            x_star: c,
            f_star: 0.0,
            metric: Metric::FunGap,
            // any ripple minimum other than the central one has f >= ~0.11
            success_tol: 1e-3,
            shgo_n: if dim == 4 { 128 } else { 512 },
        });
    }
    {
        let dim = 6;
        let c = shift(dim);
        let w = ellipsoid_weights(dim, 2.0);
        let c2 = c.clone();
        v.push(Problem {
            name: "noisy-6d",
            dim,
            bounds: vec![(-5.0, 5.0); dim],
            f: Arc::new(move |x: &[f64]| noisy_ellipsoid(x, &c2, &w)),
            x_star: c,
            f_star: 0.0,
            metric: Metric::Distance,
            // within 0.1 of the optimum: f ~ 0.01 there, far below the noise
            success_tol: 0.1,
            shgo_n: 256,
        });
    }
    {
        let dim = 5;
        let c = shift(dim);
        let w = ellipsoid_weights(dim, 2.0);
        let c2 = c.clone();
        v.push(Problem {
            name: "step-5d",
            dim,
            bounds: vec![(-5.0, 5.0); dim],
            f: Arc::new(move |x: &[f64]| step_ellipsoid(x, &c2, &w)),
            x_star: c,
            f_star: 0.0,
            metric: Metric::FunGap,
            success_tol: 1e-12, // must land on the zero plateau
            shgo_n: 256,
        });
    }
    v
}

// ---------------------------------------------------------------------------
// Optimizers under test
// ---------------------------------------------------------------------------

struct Contender {
    name: &'static str,
    algorithm: LocalOptimizer,
    /// CMA-ES population as a multiple of the crate default (ignored otherwise)
    pop_multiplier: usize,
}

fn contenders() -> Vec<Contender> {
    vec![
        Contender { name: "Bobyqa", algorithm: LocalOptimizer::Bobyqa, pop_multiplier: 1 },
        Contender { name: "Lbfgs", algorithm: LocalOptimizer::Lbfgs, pop_multiplier: 1 },
        Contender { name: "NelderMead", algorithm: LocalOptimizer::NelderMead, pop_multiplier: 1 },
        Contender { name: "Sbplx", algorithm: LocalOptimizer::Sbplx, pop_multiplier: 1 },
        Contender { name: "Cmaes", algorithm: LocalOptimizer::Cmaes, pop_multiplier: 1 },
        Contender { name: "Cmaes-4xpop", algorithm: LocalOptimizer::Cmaes, pop_multiplier: 4 },
    ]
}

/// The `cmaes` crate's default population size.
fn default_population(dim: usize) -> usize {
    4 + (3.0 * (dim as f64).ln()).floor() as usize
}

fn local_options(c: &Contender, dim: usize) -> LocalOptimizerOptions {
    LocalOptimizerOptions {
        algorithm: c.algorithm,
        ftol_rel: 1e-12,
        ftol_abs: 1e-14,
        xtol_rel: 1e-8,
        xtol_abs: 1e-14,
        maxeval: Some((1000 * dim) as u32),
        cmaes: CmaesOptions {
            population_size: if c.pop_multiplier > 1 {
                Some(c.pop_multiplier * default_population(dim))
            } else {
                None
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Part A: local optimizer alone
// ---------------------------------------------------------------------------

/// Fixed pseudo-random starting points (LCG), identical for every optimizer.
fn starting_points(p: &Problem, count: usize) -> Vec<Vec<f64>> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d ^ (p.dim as u64 * 7919);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            p.bounds
                .iter()
                .map(|&(lo, hi)| lo + 0.1 * (hi - lo) + 0.8 * (hi - lo) * next())
                .collect()
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn part_a(problems: &[Problem], starts: usize) {
    println!("## Part A: local optimizer alone ({} starts per problem)\n", starts);
    println!("| problem | optimizer | reached | median err | worst err | median nfev | time ms |");
    println!("|---|---|---|---|---|---|---|");
    for p in problems {
        let x0s = starting_points(p, starts);
        for c in contenders() {
            let opts = local_options(&c, p.dim);
            let mut errs = Vec::new();
            let mut nfevs = Vec::new();
            let mut reached = 0;
            let t = Instant::now();
            for x0 in &x0s {
                let f = Arc::clone(&p.f);
                let r = minimize_local(
                    &|x: &[f64]| f(x),
                    x0,
                    &p.bounds,
                    None::<&[fn(&[f64]) -> f64]>,
                    &opts,
                );
                let e = p.error(&r.x, r.fun);
                if e <= p.success_tol {
                    reached += 1;
                }
                errs.push(e);
                nfevs.push(r.nfev as f64);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let worst = errs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "| {} | {} | {}/{} | {:.1e} | {:.1e} | {:.0} | {:.1} |",
                p.name,
                c.name,
                reached,
                starts,
                median(&mut errs),
                worst,
                median(&mut nfevs),
                ms
            );
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Part B: through SHGO
// ---------------------------------------------------------------------------

fn part_b(problems: &[Problem]) {
    println!("## Part B: through SHGO (Sobol + k-NN, maxiter 2)\n");
    println!("| problem | optimizer | reached | err | nfev | nlfev | minima | time s |");
    println!("|---|---|---|---|---|---|---|---|");
    for p in problems {
        for c in contenders() {
            let f = Arc::clone(&p.f);
            let opts = ShgoOptions {
                sampling_method: SamplingMethod::Sobol,
                connectivity_method: ConnectivityMethod::KNearestNeighbors,
                n: p.shgo_n,
                maxiter: Some(2),
                local_options: local_options(&c, p.dim),
                disp: 0,
                ..Default::default()
            };
            let t = Instant::now();
            let r = Shgo::new(move |x: &[f64]| f(x), p.bounds.clone())
                .with_options(opts)
                .minimize()
                .unwrap();
            let secs = t.elapsed().as_secs_f64();
            let e = p.error(&r.x, r.fun);
            println!(
                "| {} | {} | {} | {:.1e} | {} | {} | {} | {:.2} |",
                p.name,
                c.name,
                if e <= p.success_tol { "yes" } else { "no" },
                e,
                r.nfev,
                r.nlfev,
                r.xl.len(),
                secs
            );
        }
    }
    println!();
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let problems = problems(quick);
    let starts = if quick { 4 } else { 8 };
    println!("# Local optimizer benchmark\n");
    println!(
        "Common settings: ftol_rel 1e-12, xtol_rel 1e-8, maxeval 1000·dim, CMA-ES sigma0 0.25 (of bound width); \
         `Cmaes-4xpop` uses four times the default population `4 + floor(3 ln dim)`.\n"
    );
    part_a(&problems, starts);
    part_b(&problems);
}
