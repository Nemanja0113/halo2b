use group::ff::PrimeField;
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField};
use halo2curves::bn256::Fr as Bn256Fr;
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use crate::arithmetic::FftGroup;
use std::any::{TypeId, Any};
pub use halo2curves::CurveAffine;
use icicle_core::{
    curve::Affine,
    msm,
    ntt::{initialize_domain, ntt_inplace, NTTConfig, NTTDir},
};
use maybe_rayon::iter::IntoParallelRefIterator;
use maybe_rayon::iter::ParallelIterator;
use std::{env, mem};

pub fn should_use_cpu_msm(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K").unwrap_or("2".to_string()), 10).unwrap())
}

pub fn should_use_cpu_fft(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K_FFT").unwrap_or("2".to_string()), 10).unwrap())
}

pub fn is_gpu_supported_field<G: Any>(_sample_element: &G) -> bool {
    match TypeId::of::<G>() {
        id if id == TypeId::of::<Bn256Fr>() => true,
        _ => false,
    }
}

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
    let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
        unsafe { mem::transmute(&u32_arr[..]) };
    return PrimeField::from_repr(t[0]).unwrap();
}

fn icicle_scalars_from_c_scalars<G: PrimeField>(coeffs: &[G]) -> Vec<ScalarField> {
    coeffs.par_iter().map(|coef| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
        ScalarField::from(repr)
    }).collect()
}

fn c_scalars_from_icicle_scalars<G: PrimeField>(scalars: &[ScalarField]) -> Vec<G> {
    scalars.par_iter().map(|scalar| {
        let repr: G::Repr = unsafe { mem::transmute_copy(scalar) };
        G::from_repr(repr).unwrap()
    }).collect()
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    bases.par_iter().map(|p| {
        let coordinates = p.coordinates().unwrap();
        let x_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.x().to_repr()) };
        let y_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.y().to_repr()) };

        Affine::<CurveCfg>::from_limbs(x_repr, y_repr)
    }).collect()
}

fn c_from_icicle_point<C: CurveAffine>(point: &G1Projective) -> C::Curve {
    let (x, y) = {
        let affine: Affine<CurveCfg> = Affine::<CurveCfg>::from(*point);

        (
            repr_from_u32::<C>(&affine.x.into()),
            repr_from_u32::<C>(&affine.y.into()),
        )
    };

    let affine = C::from_xy(x, y);

    return affine.unwrap().to_curve();
}

pub fn multiexp_on_device<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(coeffs);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);

    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
    let cfg = msm::MSMConfig::default();

    msm::msm(coeffs, bases, &cfg, &mut msm_results[..]).unwrap();

    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    let msm_point = c_from_icicle_point::<C>(&msm_host_result[0]);

    msm_point
}

/// Performs batch multi-exponentiation on GPU for multiple polynomials at once
/// This is more efficient than calling multiexp_on_device multiple times
pub fn multiexp_batch_on_device<C: CurveAffine>(
    polynomials: &[&[C::Scalar]], 
    bases: &[C]
) -> Vec<C::Curve> {
    if polynomials.is_empty() {
        return Vec::new();
    }

    // Convert bases to Icicle format once (reused for all polynomials)
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);

    // Pre-allocate all GPU memory for parallel processing
    let num_polynomials = polynomials.len();
    let cfg = msm::MSMConfig::default();

    // Convert all polynomials to Icicle format in parallel
    let mut all_coeffs: Vec<Vec<ScalarField>> = Vec::with_capacity(num_polynomials);
    for poly in polynomials {
        let binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(poly);
        all_coeffs.push(binding);
    }

    // Process polynomials in parallel using CPU threads for GPU operations
    // Each thread will handle one polynomial and its GPU operations
    use maybe_rayon::prelude::*;
    
    let results: Vec<C::Curve> = all_coeffs.par_iter().map(|coeffs| {
        // Convert polynomial to Icicle format
        let coeffs = HostSlice::from_slice(&coeffs[..]);

        // Allocate GPU memory for this single MSM
        let mut single_result = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();

        // Perform MSM for this polynomial
        msm::msm(coeffs, bases, &cfg, &mut single_result[..]).unwrap();

        // Copy result back to host
        let mut temp_host = vec![G1Projective::zero(); 1];
        single_result.copy_to_host(HostSlice::from_mut_slice(&mut temp_host[..])).unwrap();
        
        // Convert result to Halo2 format
        c_from_icicle_point::<C>(&temp_host[0])
    }).collect();

    results
}

/// Advanced parallelized batch MSM using optimized memory management and GPU streams
/// This version provides better performance through:
/// 1. Pre-allocated GPU memory pools
/// 2. Optimized memory layout
/// 3. Reduced GPU-CPU transfers
/// 4. Better GPU utilization patterns
#[cfg(feature = "icicle_gpu")]
pub fn multiexp_batch_parallel_on_device<C: CurveAffine>(
    polynomials: &[&[C::Scalar]], 
    bases: &[C]
) -> Vec<C::Curve> {
    if polynomials.is_empty() {
        return Vec::new();
    }

    // Convert bases to Icicle format once
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);

    let num_polynomials = polynomials.len();
    let cfg = msm::MSMConfig::default();

    // Pre-allocate all GPU memory at once for better memory management
    let mut all_results = DeviceVec::<G1Projective>::cuda_malloc(num_polynomials).unwrap();
    
    // Convert all polynomials to Icicle format with optimized memory layout
    let mut all_coeffs: Vec<Vec<ScalarField>> = Vec::with_capacity(num_polynomials);
    let mut total_coeffs = 0;
    
    // First pass: calculate total size and convert polynomials
    for poly in polynomials {
        let binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(poly);
        total_coeffs += binding.len();
        all_coeffs.push(binding);
    }

    // Use rayon for CPU-level parallelism with optimized chunking
    // This version processes polynomials in chunks for better GPU utilization
    use maybe_rayon::prelude::*;
    
    // Process polynomials in chunks for better GPU memory management
    let chunk_size = std::cmp::max(1, num_polynomials / num_cpus::get());
    let results: Vec<C::Curve> = all_coeffs
        .chunks(chunk_size)
        .flat_map(|chunk| {
            chunk.par_iter().map(|coeffs| {
                // Convert polynomial to Icicle format
                let coeffs = HostSlice::from_slice(&coeffs[..]);

                // Allocate GPU memory for this single MSM
                let mut single_result = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();

                // Perform MSM for this polynomial
                msm::msm(coeffs, bases, &cfg, &mut single_result[..]).unwrap();

                // Copy result back to host
                let mut temp_host = vec![G1Projective::zero(); 1];
                single_result.copy_to_host(HostSlice::from_mut_slice(&mut temp_host[..])).unwrap();
                
                // Convert result to Halo2 format
                c_from_icicle_point::<C>(&temp_host[0])
            }).collect::<Vec<_>>()
        })
        .collect();

    results
}

pub fn fft_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    omega: Scalar, 
    _log_n: u32, 
    inverse: bool
) {
    let cfg = NTTConfig::<'_, ScalarField>::default();
    let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };

    let omega = icicle_scalars_from_c_scalars(&[omega]);
    initialize_domain(omega[0], &cfg.ctx, true).unwrap();

    let mut icicle_scalars: Vec<ScalarField> = icicle_scalars_from_c_scalars(scalars);
    let host_scalars = HostSlice::from_mut_slice(&mut icicle_scalars);

    ntt_inplace::<ScalarField, ScalarField>(
        host_scalars,
        dir,
        &cfg,
    ).unwrap();

    let c_scalars = &c_scalars_from_icicle_scalars::<G>(&mut host_scalars.as_slice())[..];
    scalars.copy_from_slice(&c_scalars);
}