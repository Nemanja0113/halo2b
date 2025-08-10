//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread_local;
use std::any::TypeId;

#[cfg(feature = "icicle_gpu")]
use super::icicle;
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
    // Check if global MSM batching is enabled
    if GlobalMSMBatcher::is_batching_enabled() {
        let batch_threshold = GlobalMSMBatcher::get_batch_threshold();
        let pending_count = GlobalMSMBatcher::pending_count();
        
        // Add current operation to the batch
        let operation_id = GlobalMSMBatcher::add_operation(coeffs, bases);
        
        // If we have enough pending operations, flush the batch
        if pending_count + 1 >= batch_threshold {
            println!("🔄 [GLOBAL_MSM_BATCH] Flushing batch with {} operations", pending_count + 1);
            let completed_ids = GlobalMSMBatcher::flush_operations::<C>();
            
            // Try to get our result
            if let Some(result) = GlobalMSMBatcher::get_result::<C>(operation_id) {
                println!("   ✅ Retrieved batched result for operation {}", operation_id);
                return result;
            } else {
                println!("   ⚠️  Result not found, falling back to immediate execution");
            }
        } else {
            println!("📦 [GLOBAL_MSM_BATCH] Added operation {} to pending batch ({} total)", 
                    operation_id, pending_count + 1);
            
            // Check if we can get a result from a previous batch
            if let Some(result) = GlobalMSMBatcher::get_result::<C>(operation_id) {
                println!("   ✅ Retrieved cached result for operation {}", operation_id);
                return result;
            }
            
            // For now, execute immediately since we don't have async result handling
            println!("   ⚠️  Immediate execution (async result handling not implemented)");
        }
    }
    
    #[cfg(feature = "icicle_gpu")]
    {
        let enable_gpu = env::var("ENABLE_ICICLE_GPU").is_ok();
        let should_use_cpu = icicle::should_use_cpu_msm(coeffs.len());
        let gpu_supported = icicle::is_gpu_supported_field(&coeffs[0]);
        
        println!("🔍 [MSM_DISPATCH] MSM dispatch decision:");
        println!("   📊 Data size: {} elements", coeffs.len());
        println!("   ⚙️  ENABLE_ICICLE_GPU: {}", enable_gpu);
        println!("   🧵 Should use CPU: {}", should_use_cpu);
        println!("   🔧 GPU supported field: {}", gpu_supported);
        
        if enable_gpu && !should_use_cpu && gpu_supported {
            println!("   🚀 Using GPU MSM");
            return best_multiexp_gpu(coeffs, bases);
        } else {
            println!("   💻 Using CPU MSM");
        }
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
                coeffs.len().ilog2()
            );
        });

        // Perform MSM using Metal acceleration
        return mopro_msm::metal::msm_best::<C, H2GAffine, H2G, H2Fr>(coeffs, bases);
    }

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
    let start_time = std::time::Instant::now();
    let data_size = coeffs.len();
    
    println!("🖥️  [CPU_MSM] Starting CPU MSM operation:");
    println!("   📊 Data size: {} elements", data_size);
    println!("   🧵 Using CPU parallel processing");
    
    let msm_start = std::time::Instant::now();
    let result = msm_best(coeffs, bases);
    let msm_elapsed = msm_start.elapsed();
    
    let total_elapsed = start_time.elapsed();
    println!("✅ [CPU_MSM] CPU MSM completed in {:.2?}", total_elapsed);
    println!("   ⚡ MSM computation: {:.2?} ({:.2} elements/ms)", 
             msm_elapsed, data_size as f64 / msm_elapsed.as_millis().max(1) as f64);
    println!("   ⚡ Total throughput: {:.2} elements/ms", 
             data_size as f64 / total_elapsed.as_millis().max(1) as f64);
    
    result
}

#[cfg(feature = "icicle_gpu")]
/// Performs a multi-exponentiation operation on GPU using Icicle library
pub fn best_multiexp_gpu<C: CurveAffine>(coeffs: &[C::Scalar], g: &[C]) -> C::Curve {
    let start_time = std::time::Instant::now();
    let data_size = coeffs.len();
    let result = icicle::multiexp_on_device::<C>(coeffs, g);
    
    let elapsed = start_time.elapsed();
    
    result
}

/// Dispatcher
pub fn best_fft_cpu<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    let start_time = std::time::Instant::now();
    let data_size = a.len();
    
    println!("🖥️  [CPU_FFT] Starting CPU FFT operation:");
    println!("   📊 Data size: {} elements", data_size);
    println!("   ⚙️  Log_n: {}", log_n);
    println!("   🔄 Inverse: {}", inverse);
    
    let fft_start = std::time::Instant::now();
    fft::fft(a, omega, log_n, data, inverse);
    let fft_elapsed = fft_start.elapsed();
    
    let total_elapsed = start_time.elapsed();
    println!("✅ [CPU_FFT] CPU FFT completed in {:.2?}", total_elapsed);
    println!("   ⚡ FFT computation: {:.2?} ({:.2} elements/ms)", 
             fft_elapsed, data_size as f64 / fft_elapsed.as_millis().max(1) as f64);
    println!("   ⚡ Total throughput: {:.2} elements/ms", 
             data_size as f64 / total_elapsed.as_millis().max(1) as f64);
}

/// Best FFT
pub fn best_fft<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    #[cfg(feature = "icicle_gpu")]
    {
        let enable_gpu = env::var("ENABLE_ICICLE_GPU").is_ok();
        let should_use_cpu = icicle::should_use_cpu_fft(scalars.len());
        let gpu_supported = icicle::is_gpu_supported_field(&omega);
        
        println!("🔍 [FFT_DISPATCH] FFT dispatch decision:");
        println!("   📊 Data size: {} elements", scalars.len());
        println!("   ⚙️  ENABLE_ICICLE_GPU: {}", enable_gpu);
        println!("   🧵 Should use CPU: {}", should_use_cpu);
        println!("   🔧 GPU supported field: {}", gpu_supported);
        
        if enable_gpu && !should_use_cpu && gpu_supported {
            println!("   🚀 Using GPU FFT");
            best_fft_gpu(scalars, omega, log_n, inverse);
        } else {
            println!("   💻 Using CPU FFT");
            best_fft_cpu(scalars, omega, log_n, data, inverse);
        }
    }

    #[cfg(not(feature = "icicle_gpu"))]
    best_fft_cpu(scalars, omega, log_n, data, inverse);
}

/// Performs a NTT operation on GPU using Icicle library
#[cfg(feature = "icicle_gpu")]
pub fn best_fft_gpu<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    inverse: bool,
) {
    let start_time = std::time::Instant::now();
    let data_size = a.len();
    println!("🚀 [GPU_FFT] Starting GPU FFT operation:");
    println!("   📊 Data size: {} elements", data_size);
    println!("   ⚙️  Log_n: {}", log_n);
    println!("   🔄 Inverse: {}", inverse);
    
    icicle::fft_on_device::<Scalar, G>(a, omega, log_n, inverse);
    
    let elapsed = start_time.elapsed();
    println!("✅ [GPU_FFT] GPU FFT completed in {:.2?}", elapsed);
    println!("   ⚡ Average: {:.2} elements/ms", data_size as f64 / elapsed.as_millis().max(1) as f64);
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: PrimeCurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let start_time = std::time::Instant::now();
    let data_size = g_projective.len();
    
    println!("🖥️  [G_TO_LAGRANGE] Starting g_to_lagrange operation:");
    println!("   📊 Data size: {} elements", data_size);
    println!("   ⚙️  K: {}", k);
    println!("   🧵 Using CPU processing (curve points)");
    
    // Step 1: Setup phase
    let setup_start = std::time::Instant::now();
    let n_inv = C::Scalar::TWO_INV.pow_vartime([k as u64, 0, 0, 0]);
    let omega = C::Scalar::ROOT_OF_UNITY;
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }
    let setup_elapsed = setup_start.elapsed();
    println!("   ✅ Step 1 - Setup: {:.2?}", setup_elapsed);
    
    // Step 2: FFT computation
    let fft_start = std::time::Instant::now();
    let mut g_lagrange_projective = g_projective;
    let n = g_lagrange_projective.len();
    let fft_data = FFTData::new(n, omega, omega_inv);
    best_fft_cpu(&mut g_lagrange_projective, omega_inv, k, &fft_data, true);
    let fft_elapsed = fft_start.elapsed();
    println!("   ✅ Step 2 - FFT computation: {:.2?} ({:.2} elements/ms)", 
             fft_elapsed, data_size as f64 / fft_elapsed.as_millis().max(1) as f64);
    
    // Step 3: Scalar multiplication
    let scalar_start = std::time::Instant::now();
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });
    let scalar_elapsed = scalar_start.elapsed();
    println!("   ✅ Step 3 - Scalar multiplication: {:.2?} ({:.2} elements/ms)", 
             scalar_elapsed, data_size as f64 / scalar_elapsed.as_millis().max(1) as f64);
    
    // Step 4: Batch normalization
    let norm_start = std::time::Instant::now();
    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });
    let norm_elapsed = norm_start.elapsed();
    println!("   ✅ Step 4 - Batch normalization: {:.2?} ({:.2} elements/ms)", 
             norm_elapsed, data_size as f64 / norm_elapsed.as_millis().max(1) as f64);
    
    let total_elapsed = start_time.elapsed();
    println!("✅ [G_TO_LAGRANGE] g_to_lagrange completed in {:.2?}", total_elapsed);
    println!("   ⚡ Total throughput: {:.2} elements/ms", 
             data_size as f64 / total_elapsed.as_millis().max(1) as f64);
    println!("   📊 Breakdown:");
    println!("      - Setup: {:.1}%", 
             setup_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - FFT computation: {:.1}%", 
             fft_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - Scalar multiplication: {:.1}%", 
             scalar_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - Batch normalization: {:.1}%", 
             norm_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    
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

/// Batched FFT operations for better GPU utilization
#[cfg(feature = "icicle_gpu")]
pub fn batched_fft_operations<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    operations: &mut [(&mut [G], Scalar, u32, bool)],
) {
    if operations.is_empty() {
        return;
    }
    
    let start_time = std::time::Instant::now();
    let total_elements: usize = operations.iter().map(|(data, _, _, _)| data.len()).sum();
    
    println!("🚀 [BATCHED_FFT] Starting batched FFT operations:");
    println!("   📊 Total operations: {}", operations.len());
    println!("   📊 Total elements: {}", total_elements);
    
    #[cfg(feature = "icicle_gpu")]
    {
        let enable_gpu = env::var("ENABLE_ICICLE_GPU").is_ok();
        let gpu_supported = operations.iter().any(|(data, omega, _, _)| {
            data.len() > 0 && icicle::is_gpu_supported_field(omega)
        });
        
        if enable_gpu && gpu_supported {
            println!("   🚀 Using GPU batched FFT");
            
            // Process operations in parallel batches for better GPU utilization
            let batch_size = std::env::var("HALO2_FFT_BATCH_SIZE")
                .unwrap_or_else(|_| "4".to_string())
                .parse::<usize>()
                .unwrap_or(4);
            
            for (batch_idx, batch) in operations.chunks_mut(batch_size).enumerate() {
                let batch_start = std::time::Instant::now();
                
                // Process each operation in the batch
                for (data, omega, log_n, inverse) in batch.iter_mut() {
                    best_fft_gpu(*data, *omega, *log_n, *inverse);
                }
                
                let batch_elapsed = batch_start.elapsed();
                println!("   📦 GPU FFT batch {}: {} operations in {:.2?}", batch_idx, batch.len(), batch_elapsed);
            }
            
            let elapsed = start_time.elapsed();
            println!("✅ [BATCHED_FFT] GPU batched FFT completed in {:.2?}", elapsed);
            println!("   ⚡ Average: {:.2} operations/ms", operations.len() as f64 / elapsed.as_millis().max(1) as f64);
            return;
        }
    }
    
    // Fallback to CPU batched processing
    println!("   💻 Using CPU batched FFT");
    
    // Process operations in parallel batches for better CPU utilization
    let batch_size = std::env::var("HALO2_FFT_BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse::<usize>()
        .unwrap_or(4);
    
    for (batch_idx, batch) in operations.chunks_mut(batch_size).enumerate() {
        let batch_start = std::time::Instant::now();
        
        // Process each operation in the batch
        for (data, omega, log_n, inverse) in batch.iter_mut() {
            let fft_data = FFTData::new(data.len(), *omega, omega.invert().unwrap());
            best_fft_cpu(*data, *omega, *log_n, &fft_data, *inverse);
        }
        
        let batch_elapsed = batch_start.elapsed();
        println!("   📦 CPU FFT batch {}: {} operations in {:.2?}", batch_idx, batch.len(), batch_elapsed);
    }
    
    let elapsed = start_time.elapsed();
    println!("✅ [BATCHED_FFT] CPU batched FFT completed in {:.2?}", elapsed);
    println!("   ⚡ Average: {:.2} operations/ms", operations.len() as f64 / elapsed.as_millis().max(1) as f64);
}

/// Batched MSM operations for better GPU utilization
#[cfg(feature = "icicle_gpu")]
pub fn batched_msm_operations<C: CurveAffine>(
    operations: &[(&[C::Scalar], &[C])],
) -> Vec<C::Curve> {
    if operations.is_empty() {
        return Vec::new();
    }
    
    let start_time = std::time::Instant::now();
    let total_elements: usize = operations.iter().map(|(coeffs, _)| coeffs.len()).sum();
    
    println!("🚀 [BATCHED_MSM] Starting batched MSM operations:");
    println!("   📊 Total operations: {}", operations.len());
    println!("   📊 Total elements: {}", total_elements);
    
    #[cfg(feature = "icicle_gpu")]
    {
        let enable_gpu = env::var("ENABLE_ICICLE_GPU").is_ok();
        let enable_batching = env::var("HALO2_MSM_BATCHING")
            .unwrap_or_else(|_| "1".to_string())
            .parse::<bool>()
            .unwrap_or(true);
        let gpu_supported = operations.iter().any(|(coeffs, _)| {
            coeffs.len() > 0 && icicle::is_gpu_supported_field(&coeffs[0])
        });
        
        println!("🔍 [BATCHED_MSM] GPU dispatch decision:");
        println!("   ⚙️  ENABLE_ICICLE_GPU: {}", enable_gpu);
        println!("   ⚙️  HALO2_MSM_BATCHING: {}", enable_batching);
        println!("   🔧 GPU supported field: {}", gpu_supported);
        
        if enable_gpu && enable_batching && gpu_supported {
            println!("   🚀 Using GPU batched MSM");
            let results = icicle::batched_multiexp_on_device::<C>(operations);
            
            let elapsed = start_time.elapsed();
            println!("✅ [BATCHED_MSM] GPU batched MSM completed in {:.2?}", elapsed);
            println!("   ⚡ Average: {:.2} operations/ms", operations.len() as f64 / elapsed.as_millis().max(1) as f64);
            
            return results;
        }
    }
    
    // Fallback to CPU batched processing
    println!("   💻 Using CPU batched MSM");
    
    // Process operations in parallel batches for better CPU utilization
    let batch_size = std::env::var("HALO2_MSM_BATCH_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse::<usize>()
        .unwrap_or(4);
    
    let mut results = Vec::with_capacity(operations.len());
    
    for (batch_idx, batch) in operations.chunks(batch_size).enumerate() {
        let batch_start = std::time::Instant::now();
        
        // Process each operation in the batch
        for (coeffs, bases) in batch {
            let result = best_multiexp_cpu(coeffs, bases);
            results.push(result);
        }
        
        let batch_elapsed = batch_start.elapsed();
        println!("   📦 MSM batch {}: {} operations in {:.2?}", batch_idx, batch.len(), batch_elapsed);
    }
    
    let elapsed = start_time.elapsed();
    println!("✅ [BATCHED_MSM] CPU batched MSM completed in {:.2?}", elapsed);
    println!("   ⚡ Average: {:.2} operations/ms", operations.len() as f64 / elapsed.as_millis().max(1) as f64);
    
    results
}

/// Optimized FFT dispatch with better batching logic
pub fn optimized_fft<Scalar: Field + ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    #[cfg(feature = "icicle_gpu")]
    {
        let enable_gpu = env::var("ENABLE_ICICLE_GPU").is_ok();
        let should_use_cpu = icicle::should_use_cpu_fft(scalars.len());
        let gpu_supported = icicle::is_gpu_supported_field(&omega);
        
        // Optimized threshold: use GPU for larger operations
        let optimized_threshold = env::var("HALO2_FFT_GPU_THRESHOLD")
            .unwrap_or_else(|_| "1024".to_string())
            .parse::<usize>()
            .unwrap_or(1024);
        
        let should_use_gpu = scalars.len() >= optimized_threshold;
        
        println!("🔍 [OPTIMIZED_FFT] FFT dispatch decision:");
        println!("   📊 Data size: {} elements", scalars.len());
        println!("   ⚙️  ENABLE_ICICLE_GPU: {}", enable_gpu);
        println!("   🧵 Should use CPU (original): {}", should_use_cpu);
        println!("   🚀 Should use GPU (optimized): {}", should_use_gpu);
        println!("   🔧 GPU supported field: {}", gpu_supported);
        
        if enable_gpu && should_use_gpu && gpu_supported {
            println!("   🚀 Using GPU FFT (optimized)");
            best_fft_gpu(scalars, omega, log_n, inverse);
        } else {
            println!("   💻 Using CPU FFT");
            best_fft_cpu(scalars, omega, log_n, data, inverse);
        }
    }

    #[cfg(not(feature = "icicle_gpu"))]
    best_fft_cpu(scalars, omega, log_n, data, inverse);
}

/// Optimized MSM dispatch with better batching logic
pub fn optimized_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    #[cfg(feature = "icicle_gpu")]
    {
        let enable_gpu = env::var("ENABLE_ICICLE_GPU").is_ok();
        let should_use_cpu = icicle::should_use_cpu_msm(coeffs.len());
        let gpu_supported = icicle::is_gpu_supported_field(&coeffs[0]);
        
        // Optimized threshold: use GPU for larger operations
        let optimized_threshold = env::var("HALO2_MSM_GPU_THRESHOLD")
            .unwrap_or_else(|_| "512".to_string())
            .parse::<usize>()
            .unwrap_or(512);
        
        let should_use_gpu = coeffs.len() >= optimized_threshold;
        
        println!("🔍 [OPTIMIZED_MSM] MSM dispatch decision:");
        println!("   📊 Data size: {} elements", coeffs.len());
        println!("   ⚙️  ENABLE_ICICLE_GPU: {}", enable_gpu);
        println!("   🧵 Should use CPU (original): {}", should_use_cpu);
        println!("   🚀 Should use GPU (optimized): {}", should_use_gpu);
        println!("   🔧 GPU supported field: {}", gpu_supported);
        
        if enable_gpu && should_use_gpu && gpu_supported {
            println!("   🚀 Using GPU MSM (optimized)");
            return best_multiexp_gpu(coeffs, bases);
        } else {
            println!("   💻 Using CPU MSM");
        }
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
                coeffs.len().ilog2()
            );
        });

        // Perform MSM using Metal acceleration
        return mopro_msm::metal::msm_best::<C, H2GAffine, H2G, H2Fr>(coeffs, bases);
    }

    #[allow(unreachable_code)]
    best_multiexp_cpu(coeffs, bases)
}

// Global MSM batching system
thread_local! {
    static PENDING_MSM_OPERATIONS: std::cell::RefCell<Vec<MSMOperation>> = std::cell::RefCell::new(Vec::new());
    static MSM_RESULTS: std::cell::RefCell<HashMap<usize, MSMResult>> = std::cell::RefCell::new(HashMap::new());
    static MSM_BATCH_ID: std::cell::RefCell<usize> = std::cell::RefCell::new(0);
}

static GLOBAL_MSM_BATCH_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
struct MSMOperation {
    id: usize,
    coeffs: Vec<u8>, // Serialized coefficients
    bases: Vec<u8>,  // Serialized bases
    curve_type: std::any::TypeId,
    size: usize,
}

#[derive(Debug, Clone)]
enum MSMResult {
    Pending,
    Completed(Vec<u8>), // Serialized result
}

impl MSMOperation {
    fn new<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> Self {
        // Serialize the data for storage
        let coeffs_bytes = unsafe {
            std::slice::from_raw_parts(
                coeffs.as_ptr() as *const u8,
                coeffs.len() * std::mem::size_of::<C::Scalar>(),
            )
        }.to_vec();
        
        let bases_bytes = unsafe {
            std::slice::from_raw_parts(
                bases.as_ptr() as *const u8,
                bases.len() * std::mem::size_of::<C>(),
            )
        }.to_vec();
        
        Self {
            id: GLOBAL_MSM_BATCH_COUNTER.fetch_add(1, Ordering::Relaxed),
            coeffs: coeffs_bytes,
            bases: bases_bytes,
            curve_type: std::any::TypeId::of::<C>(),
            size: coeffs.len(),
        }
    }
}

/// Global MSM batching context
pub struct GlobalMSMBatcher;

impl GlobalMSMBatcher {
    /// Add an MSM operation to the pending batch
    pub fn add_operation<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> usize {
        let operation = MSMOperation::new(coeffs, bases);
        let id = operation.id;
        
        PENDING_MSM_OPERATIONS.with(|pending| {
            pending.borrow_mut().push(operation);
        });
        
        // Initialize result as pending
        MSM_RESULTS.with(|results| {
            results.borrow_mut().insert(id, MSMResult::Pending);
        });
        
        id
    }
    
    /// Get result for a specific operation ID
    pub fn get_result<C: CurveAffine>(operation_id: usize) -> Option<C::Curve> {
        MSM_RESULTS.with(|results| {
            let mut results_map = results.borrow_mut();
            if let Some(result) = results_map.get(&operation_id) {
                match result {
                    MSMResult::Completed(bytes) => {
                        // Deserialize the result
                        let curve: C::Curve = unsafe {
                            std::ptr::read(bytes.as_ptr() as *const C::Curve)
                        };
                        Some(curve)
                    }
                    MSMResult::Pending => None,
                }
            } else {
                None
            }
        })
    }
    
    /// Flush all pending MSM operations and store results
    pub fn flush_operations<C: CurveAffine>() -> Vec<usize> {
        let operations = PENDING_MSM_OPERATIONS.with(|pending| {
            let mut ops = pending.borrow_mut();
            // Use stable alternative to drain_filter
            let mut filtered_ops = Vec::new();
            let mut i = 0;
            while i < ops.len() {
                if std::any::TypeId::of::<C>() == ops[i].curve_type {
                    filtered_ops.push(ops.remove(i));
                } else {
                    i += 1;
                }
            }
            filtered_ops
        });
        
        if operations.is_empty() {
            return Vec::new();
        }
        
        // Convert back to the proper format for batched processing
        let mut batched_ops = Vec::new();
        let mut operation_ids = Vec::new();
        
        for op in operations {
            let coeffs: &[C::Scalar] = unsafe {
                std::slice::from_raw_parts(
                    op.coeffs.as_ptr() as *const C::Scalar,
                    op.coeffs.len() / std::mem::size_of::<C::Scalar>(),
                )
            };
            
            let bases: &[C] = unsafe {
                std::slice::from_raw_parts(
                    op.bases.as_ptr() as *const C,
                    op.bases.len() / std::mem::size_of::<C>(),
                )
            };
            
            batched_ops.push((coeffs, bases));
            operation_ids.push(op.id);
        }
        
        // Use the existing batched MSM function
        let results = batched_msm_operations(&batched_ops);
        
        // Store results in the results map
        MSM_RESULTS.with(|results_map| {
            let mut map = results_map.borrow_mut();
            for (id, result) in operation_ids.iter().zip(results.iter()) {
                let result_bytes = unsafe {
                    std::slice::from_raw_parts(
                        result as *const C::Curve as *const u8,
                        std::mem::size_of::<C::Curve>(),
                    )
                }.to_vec();
                map.insert(*id, MSMResult::Completed(result_bytes));
            }
        });
        
        operation_ids
    }
    
    /// Check if batching is enabled
    pub fn is_batching_enabled() -> bool {
        env::var("HALO2_GLOBAL_MSM_BATCHING")
            .unwrap_or_else(|_| "1".to_string())
            .parse::<bool>()
            .unwrap_or(true)
    }
    
    /// Get the current batch size threshold
    pub fn get_batch_threshold() -> usize {
        env::var("HALO2_GLOBAL_MSM_BATCH_SIZE")
            .unwrap_or_else(|_| "4".to_string())
            .parse::<usize>()
            .unwrap_or(4)
    }
    
    /// Get the number of pending operations
    pub fn pending_count() -> usize {
        PENDING_MSM_OPERATIONS.with(|pending| pending.borrow().len())
    }
    
    /// Force flush all remaining operations (call at end of proof generation)
    pub fn force_flush_all<C: CurveAffine>() {
        let pending_count = Self::pending_count();
        if pending_count > 0 {
            println!("🔄 [GLOBAL_MSM_BATCH] Force flushing {} remaining operations", pending_count);
            Self::flush_operations::<C>();
        }
    }
    
    /// Clear all pending operations and results (for cleanup)
    pub fn clear_all() {
        PENDING_MSM_OPERATIONS.with(|pending| {
            pending.borrow_mut().clear();
        });
        MSM_RESULTS.with(|results| {
            results.borrow_mut().clear();
        });
        println!("🧹 [GLOBAL_MSM_BATCH] Cleared all pending operations and results");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::{Fr, G1Affine};
    use halo2curves::CurveAffine;
    
    #[test]
    fn test_global_msm_batching() {
        // Set environment variables for testing
        std::env::set_var("HALO2_GLOBAL_MSM_BATCHING", "1");
        std::env::set_var("HALO2_GLOBAL_MSM_BATCH_SIZE", "2");
        std::env::set_var("ENABLE_ICICLE_GPU", "false"); // Use CPU for testing
        
        // Clear any existing state
        GlobalMSMBatcher::clear_all();
        
        // Create test data
        let coeffs1 = vec![Fr::from(1u64), Fr::from(2u64)];
        let bases1 = vec![G1Affine::generator(), G1Affine::generator()];
        
        let coeffs2 = vec![Fr::from(3u64), Fr::from(4u64)];
        let bases2 = vec![G1Affine::generator(), G1Affine::generator()];
        
        // First operation should be added to pending batch
        let result1 = best_multiexp::<G1Affine>(&coeffs1, &bases1);
        
        // Second operation should trigger batch flush
        let result2 = best_multiexp::<G1Affine>(&coeffs2, &bases2);
        
        // Verify results are not zero
        assert!(!result1.is_identity().into());
        assert!(!result2.is_identity().into());
        
        // Force flush any remaining operations
        GlobalMSMBatcher::force_flush_all::<G1Affine>();
        
        // Clear state
        GlobalMSMBatcher::clear_all();
        
        println!("✅ Global MSM batching test passed");
    }
}
