//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

#[cfg(feature = "icicle_gpu")]
use super::icicle;
#[cfg(feature = "icicle_gpu")]
use std::env;
use super::multicore;
pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    prime::PrimeCurveAffine,
    Curve, GroupOpsOwned, ScalarMulOwned,
};

use halo2curves::msm::msm_best;
pub use halo2curves::{CurveAffine, CurveExt};

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

/// Best MSM
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let data_size = coeffs.len();
    let log_size = data_size.ilog2();
    
    #[cfg(feature = "icicle_gpu")]
    if env::var("ENABLE_ICICLE_GPU").is_ok()
        && !icicle::should_use_cpu_msm(coeffs.len())
        && icicle::is_gpu_supported_field(&coeffs[0])
    {
        log::debug!("🔄 [MSM_DISPATCH] Using GPU path: {} elements (log2: {})", data_size, log_size);
        return best_multiexp_gpu(coeffs, bases);
    }

    #[cfg(feature = "metal")]
    {
        use mopro_msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
        use std::sync::Once;

        // Static mutex to block concurrent Metal acceleration calls
        static PRINT_ONCE: Once = Once::new();

        // Print the warning message only once
        PRINT_ONCE.call_once(|| {
            log::warn!(
                "WARNING: Using Experimental Metal Acceleration for MSM. \
                 Best performance improvements are observed with log row size >= 20. \
                 Current log size: {}",
                log_size
            );
        });

        log::debug!("🔄 [MSM_DISPATCH] Using Metal path: {} elements (log2: {})", data_size, log_size);
        
        // Perform MSM using Metal acceleration with timing
        use instant::Instant;
        let msm_start = Instant::now();
        let result = mopro_msm::metal::msm_best::<C, H2GAffine, H2G, H2Fr>(coeffs, bases);
        let elapsed = msm_start.elapsed();
        
        log::info!("⚡ [MSM_METAL] Metal MSM completed: {} elements in {:?} ({:.2} elements/ms)", 
                   data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
        
        // Update global counters
        MSM_COUNTER.fetch_add(1, Ordering::Relaxed);
        MSM_METAL_COUNTER.fetch_add(1, Ordering::Relaxed);
        *MSM_TOTAL_TIME.lock().unwrap() += elapsed;
        
        return result;
    }

    log::debug!("🔄 [MSM_DISPATCH] Using CPU path: {} elements (log2: {})", data_size, log_size);
    #[allow(unreachable_code)]
    best_multiexp_cpu(coeffs, bases)
}

// [JPW] Keep this adapter to halo2curves to minimize code changes.
/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp_cpu<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    use instant::Instant;
    
    let msm_start = Instant::now();
    let data_size = coeffs.len();
    
    log::debug!("🚀 [MSM_CPU] Starting CPU MSM: {} elements", data_size);
    
    let result = msm_best(coeffs, bases);
    
    let elapsed = msm_start.elapsed();
    log::info!("⚡ [MSM_CPU] CPU MSM completed: {} elements in {:?} ({:.2} elements/ms)", 
               data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
    
    // Update global counters
    MSM_COUNTER.fetch_add(1, Ordering::Relaxed);
    MSM_CPU_COUNTER.fetch_add(1, Ordering::Relaxed);
    *MSM_TOTAL_TIME.lock().unwrap() += elapsed;
    
    result
}

#[cfg(feature = "icicle_gpu")]
/// Performs a multi-exponentiation operation on GPU using Icicle library
pub fn best_multiexp_gpu<C: CurveAffine>(coeffs: &[C::Scalar], g: &[C]) -> C::Curve {
    use instant::Instant;
    
    let msm_start = Instant::now();
    let data_size = coeffs.len();
    
    log::debug!("🚀 [MSM_GPU] Starting GPU MSM: {} elements", data_size);
    
    let result = icicle::multiexp_on_device::<C>(coeffs, g);
    
    let elapsed = msm_start.elapsed();
    log::info!("⚡ [MSM_GPU] GPU MSM completed: {} elements in {:?} ({:.2} elements/ms)", 
               data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
    
    // Update global counters
    MSM_COUNTER.fetch_add(1, Ordering::Relaxed);
    MSM_GPU_COUNTER.fetch_add(1, Ordering::Relaxed);
    *MSM_TOTAL_TIME.lock().unwrap() += elapsed;
    
    result
}

/// Batch MSM structure to hold multiple polynomials and their blinding factors
#[derive(Debug)]
pub struct BatchMSMInput<C: CurveAffine> {
    pub polynomials: Vec<Vec<C::Scalar>>,
    pub blinding_factors: Vec<C::Scalar>,
}

impl<C: CurveAffine> BatchMSMInput<C> {
    pub fn new() -> Self {
        Self {
            polynomials: Vec::new(),
            blinding_factors: Vec::new(),
        }
    }

    pub fn add_polynomial(&mut self, polynomial: &[C::Scalar], blinding_factor: C::Scalar) {
        self.polynomials.push(polynomial.to_vec());
        self.blinding_factors.push(blinding_factor);
    }

    pub fn is_empty(&self) -> bool {
        self.polynomials.is_empty()
    }

    pub fn len(&self) -> usize {
        self.polynomials.len()
    }

    /// Flatten all polynomials into a single vector for batch processing
    pub fn flatten(&self) -> (Vec<C::Scalar>, Vec<usize>) {
        let mut flattened = Vec::new();
        let mut offsets = Vec::new();
        let mut current_offset = 0;

        for poly in &self.polynomials {
            offsets.push(current_offset);
            flattened.extend(poly.iter());
            current_offset += poly.len();
        }

        (flattened, offsets)
    }
}

/// Performs batch multi-exponentiation operations
/// This function will use GPU if available and beneficial
pub fn best_batch_multiexp<C: CurveAffine>(
    batch_input: &BatchMSMInput<C>,
    bases: &[C],
) -> Vec<C::Curve> {
    if batch_input.is_empty() {
        return Vec::new();
    }

    let data_size: usize = batch_input.polynomials.iter().map(|p| p.len()).sum();
    let log_size = (data_size as f64).log2().ceil() as u32;

    #[cfg(feature = "icicle_gpu")]
    if env::var("ENABLE_ICICLE_GPU").is_ok()
        && !icicle::should_use_cpu_msm(data_size)
        && icicle::is_gpu_supported_field(&batch_input.polynomials[0][0])
    {
        log::debug!("🔄 [BATCH_MSM_DISPATCH] Using GPU path: {} polynomials, {} total elements", 
                   batch_input.len(), data_size);
        return best_batch_multiexp_gpu(batch_input, bases);
    }

    log::debug!("🔄 [BATCH_MSM_DISPATCH] Using CPU path: {} polynomials, {} total elements", 
               batch_input.len(), data_size);
    best_batch_multiexp_cpu(batch_input, bases)
}

/// Performs batch multi-exponentiation operations on CPU
pub fn best_batch_multiexp_cpu<C: CurveAffine>(
    batch_input: &BatchMSMInput<C>,
    bases: &[C],
) -> Vec<C::Curve> {
    use instant::Instant;
    
    let msm_start = Instant::now();
    let data_size: usize = batch_input.polynomials.iter().map(|p| p.len()).sum();
    
    log::debug!("🚀 [BATCH_MSM_CPU] Starting CPU batch MSM: {} polynomials, {} total elements", 
               batch_input.len(), data_size);
    
    let mut results = Vec::with_capacity(batch_input.len());
    
    // Process each polynomial individually (CPU fallback)
    for poly in &batch_input.polynomials {
        let size = poly.len();
        assert!(bases.len() >= size);
        
        let result = msm_best(poly, &bases[0..size]);
        results.push(result);
    }
    
    let elapsed = msm_start.elapsed();
    log::info!("⚡ [BATCH_MSM_CPU] CPU batch MSM completed: {} polynomials, {} total elements in {:?} ({:.2} elements/ms)", 
               batch_input.len(), data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
    
    // Update global counters
    MSM_COUNTER.fetch_add(batch_input.len() as usize, Ordering::Relaxed);
    MSM_CPU_COUNTER.fetch_add(batch_input.len() as usize, Ordering::Relaxed);
    *MSM_TOTAL_TIME.lock().unwrap() += elapsed;
    
    results
}

#[cfg(feature = "icicle_gpu")]
/// Performs batch multi-exponentiation operations on GPU using Icicle library
pub fn best_batch_multiexp_gpu<C: CurveAffine>(
    batch_input: &BatchMSMInput<C>,
    bases: &[C],
) -> Vec<C::Curve> {
    use instant::Instant;
    
    let msm_start = Instant::now();
    let data_size: usize = batch_input.polynomials.iter().map(|p| p.len()).sum();
    
    log::debug!("🚀 [BATCH_MSM_GPU] Starting parallelized GPU batch MSM: {} polynomials, {} total elements", 
               batch_input.len(), data_size);
    
    // Convert polynomials to slice references for the batch function
    let polynomials: Vec<&[C::Scalar]> = batch_input.polynomials.iter().map(|p| p.as_slice()).collect();
    
    // Use the new parallelized batch MSM function
    // let results = icicle::multiexp_batch_on_device::<C>(&polynomials, bases);
    let results = icicle::multiexp_batch_parallel_on_device::<C>(&polynomials, bases);
    
    let elapsed = msm_start.elapsed();
    log::info!("⚡ [BATCH_MSM_GPU] Parallelized GPU batch MSM completed: {} polynomials, {} total elements in {:?} ({:.2} elements/ms)", 
               batch_input.len(), data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
    
    // Update global counters
    MSM_COUNTER.fetch_add(batch_input.len() as usize, Ordering::Relaxed);
    MSM_GPU_COUNTER.fetch_add(batch_input.len() as usize, Ordering::Relaxed);
    *MSM_TOTAL_TIME.lock().unwrap() += elapsed;
    
    results
}

/// Dispatcher
pub fn best_fft_cpu<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    use instant::Instant;
    let fft_start = Instant::now();
    let data_size = a.len();
    log::debug!("🚀 [FFT_CPU] Starting CPU FFT: {} elements (log2: {})", data_size, log_n);
    
    fft::fft(a, omega, log_n, data, inverse);
    
    let elapsed = fft_start.elapsed();
    log::info!("⚡ [FFT_CPU] CPU FFT completed: {} elements in {:?} ({:.2} elements/ms)",
               data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
    
    // Update global counters
    FFT_COUNTER.fetch_add(1, Ordering::Relaxed);
    FFT_CPU_COUNTER.fetch_add(1, Ordering::Relaxed);
    *FFT_TOTAL_TIME.lock().unwrap() += elapsed;
}

/// Best FFT
pub fn best_fft<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    let data_size = scalars.len();
    
    #[cfg(feature = "icicle_gpu")]
    if env::var("ENABLE_ICICLE_GPU").is_ok()
        && !icicle::should_use_cpu_fft(scalars.len())
        && icicle::is_gpu_supported_field(&omega)
    {
        log::debug!("🔄 [FFT_DISPATCH] Using GPU path: {} elements (log2: {})", data_size, log_n);
        best_fft_gpu(scalars, omega, log_n, inverse);
    } else {
        log::debug!("🔄 [FFT_DISPATCH] Using CPU path: {} elements (log2: {})", data_size, log_n);
        best_fft_cpu(scalars, omega, log_n, data, inverse);
    }

    #[cfg(not(feature = "icicle_gpu"))]
    {
        log::debug!("🔄 [FFT_DISPATCH] Using CPU path: {} elements (log2: {})", data_size, log_n);
        best_fft_cpu(scalars, omega, log_n, data, inverse);
    }
}

/// Performs a NTT operation on GPU using Icicle library
#[cfg(feature = "icicle_gpu")]
pub fn best_fft_gpu<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    inverse: bool,
) {
    use instant::Instant;
    let fft_start = Instant::now();
    let data_size = a.len();
    log::debug!("🚀 [FFT_GPU] Starting GPU FFT: {} elements (log2: {})", data_size, log_n);
    
    icicle::fft_on_device::<Scalar, G>(a, omega, log_n, inverse);
    
    let elapsed = fft_start.elapsed();
    log::info!("⚡ [FFT_GPU] GPU FFT completed: {} elements in {:?} ({:.2} elements/ms)",
               data_size, elapsed, data_size as f64 / elapsed.as_millis() as f64);
    
    // Update global counters
    FFT_COUNTER.fetch_add(1, Ordering::Relaxed);
    FFT_GPU_COUNTER.fetch_add(1, Ordering::Relaxed);
    *FFT_TOTAL_TIME.lock().unwrap() += elapsed;
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: PrimeCurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let n_inv = C::Scalar::TWO_INV.pow_vartime([k as u64, 0, 0, 0]);
    let omega = C::Scalar::ROOT_OF_UNITY;
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }

    let mut g_lagrange_projective = g_projective;
    let n = g_lagrange_projective.len();
    let fft_data = FFTData::new(n, omega, omega_inv);

    best_fft_cpu(&mut g_lagrange_projective, omega_inv, k, &fft_data, true);
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });

    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });

    g_lagrange
}
/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::ZERO, |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ZERO; num_threads];
        multicore::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime([start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::ZERO, |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    // TODO: parallelize?
    assert_eq!(a.len(), b.len());

    let mut acc = F::ZERO;
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - 1];

    let mut tmp = F::ZERO;
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// This utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    // Algorithm rationale:
    //
    // Using the stdlib `chunks_mut` will lead to severe load imbalance.
    // From https://github.com/rust-lang/rust/blob/e94bda3/library/core/src/slice/iter.rs#L1607-L1637
    // if the division is not exact, the last chunk will be the remainder.
    //
    // Dividing 40 items on 12 threads will lead to a chunk size of 40/12 = 3,
    // There will be a 13 chunks of size 3 and 1 of size 1 distributed on 12 threads.
    // This leads to 1 thread working on 6 iterations, 1 on 4 iterations and 10 on 3 iterations,
    // a load imbalance of 2x.
    //
    // Instead we can divide work into chunks of size
    // 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3 = 4*4 + 3*8 = 40
    //
    // This would lead to a 6/4 = 1.5x speedup compared to naive chunks_mut
    //
    // See also OpenMP spec (page 60)
    // http://www.openmp.org/mp-documents/openmp-4.5.pdf
    // "When no chunk_size is specified, the iteration space is divided into chunks
    // that are approximately equal in size, and at most one chunk is distributed to
    // each thread. The size of the chunks is unspecified in this case."
    // This implies chunks are the same size ±1

    let f = &f;
    let total_iters = v.len();
    let num_threads = multicore::current_num_threads();
    let base_chunk_size = total_iters / num_threads;
    let cutoff_chunk_id = total_iters % num_threads;
    let split_pos = cutoff_chunk_id * (base_chunk_size + 1);
    let (v_hi, v_lo) = v.split_at_mut(split_pos);

    multicore::scope(|scope| {
        // Skip special-case: number of iterations is cleanly divided by number of threads.
        if cutoff_chunk_id != 0 {
            for (chunk_id, chunk) in v_hi.chunks_exact_mut(base_chunk_size + 1).enumerate() {
                let offset = chunk_id * (base_chunk_size + 1);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
        // Skip special-case: less iterations than number of threads.
        if base_chunk_size != 0 {
            for (chunk_id, chunk) in v_lo.chunks_exact_mut(base_chunk_size).enumerate() {
                let offset = split_pos + (chunk_id * base_chunk_size);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
    });
}

///
pub fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms)
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: Field>(roots: &[F], z: F) -> F {
    fn evaluate<F: Field>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::ONE, |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ONE; num_threads];
        multicore::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::ONE, |acc, part| acc * part)
    }
}

pub(crate) fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::ONE), move |power| Some(base * power))
}

/// Reverse `l` LSBs of bitvector `n`
pub fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[cfg(test)]
use rand_core::OsRng;

use crate::fft::{self, recursive::FFTData};
#[cfg(test)]
use crate::halo2curves::pasta::Fp;
// use crate::plonk::{get_duration, get_time, start_measure, stop_measure};

// Global MSM tracking
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

lazy_static::lazy_static! {
    static ref MSM_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static ref MSM_TOTAL_TIME: Mutex<Duration> = Mutex::new(Duration::ZERO);
    static ref MSM_GPU_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static ref MSM_CPU_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static ref MSM_METAL_COUNTER: AtomicUsize = AtomicUsize::new(0);
    
    // FFT statistics tracking
    static ref FFT_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static ref FFT_TOTAL_TIME: Mutex<Duration> = Mutex::new(Duration::ZERO);
    static ref FFT_GPU_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static ref FFT_CPU_COUNTER: AtomicUsize = AtomicUsize::new(0);
}

pub fn get_msm_stats() -> (usize, Duration, usize, usize, usize) {
    let total_count = MSM_COUNTER.load(Ordering::Relaxed);
    let total_time = *MSM_TOTAL_TIME.lock().unwrap();
    let gpu_count = MSM_GPU_COUNTER.load(Ordering::Relaxed);
    let cpu_count = MSM_CPU_COUNTER.load(Ordering::Relaxed);
    let metal_count = MSM_METAL_COUNTER.load(Ordering::Relaxed);
    (total_count, total_time, gpu_count, cpu_count, metal_count)
}

pub fn reset_msm_stats() {
    MSM_COUNTER.store(0, Ordering::Relaxed);
    *MSM_TOTAL_TIME.lock().unwrap() = Duration::ZERO;
    MSM_GPU_COUNTER.store(0, Ordering::Relaxed);
    MSM_CPU_COUNTER.store(0, Ordering::Relaxed);
    MSM_METAL_COUNTER.store(0, Ordering::Relaxed);
}

pub fn get_fft_stats() -> (usize, Duration, usize, usize) {
    let total_count = FFT_COUNTER.load(Ordering::Relaxed);
    let total_time = *FFT_TOTAL_TIME.lock().unwrap();
    let gpu_count = FFT_GPU_COUNTER.load(Ordering::Relaxed);
    let cpu_count = FFT_CPU_COUNTER.load(Ordering::Relaxed);
    (total_count, total_time, gpu_count, cpu_count)
}

pub fn reset_fft_stats() {
    FFT_COUNTER.store(0, Ordering::Relaxed);
    *FFT_TOTAL_TIME.lock().unwrap() = Duration::ZERO;
    FFT_GPU_COUNTER.store(0, Ordering::Relaxed);
    FFT_CPU_COUNTER.store(0, Ordering::Relaxed);
}

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::{Bn256, Fr};
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_batch_msm_input() {
        let mut batch_input = BatchMSMInput::<Bn256>::new();
        
        // Create test polynomials
        let poly1 = vec![Fr::random(OsRng), Fr::random(OsRng), Fr::random(OsRng)];
        let poly2 = vec![Fr::random(OsRng), Fr::random(OsRng)];
        
        let blind1 = Fr::random(OsRng);
        let blind2 = Fr::random(OsRng);
        
        batch_input.add_polynomial(&poly1, blind1);
        batch_input.add_polynomial(&poly2, blind2);
        
        assert_eq!(batch_input.len(), 2);
        assert!(!batch_input.is_empty());
        
        let (flattened, offsets) = batch_input.flatten();
        assert_eq!(flattened.len(), 5); // 3 + 2
        assert_eq!(offsets, vec![0, 3]);
    }

    #[test]
    fn test_batch_msm_cpu() {
        let mut batch_input = BatchMSMInput::<Bn256>::new();
        
        // Create test polynomials
        let poly1 = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let poly2 = vec![Fr::from(4), Fr::from(5)];
        
        let blind1 = Fr::from(10);
        let blind2 = Fr::from(20);
        
        batch_input.add_polynomial(&poly1, blind1);
        batch_input.add_polynomial(&poly2, blind2);
        
        // Create dummy bases (this is just for testing the structure)
        let bases = vec![Bn256::generator(); 3];
        
        let results = best_batch_multiexp_cpu(&batch_input, &bases);
        assert_eq!(results.len(), 2);
    }
}
