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
use maybe_rayon::iter::IndexedParallelIterator;
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

/// Performs batched multi-exponentiation operations on GPU using Icicle library
/// This function computes multiple MSM operations in parallel for better GPU utilization
pub fn batched_multiexp_on_device<C: CurveAffine>(
    coeffs_batches: &[&[C::Scalar]], 
    bases_batches: &[&[C]]
) -> Vec<C::Curve> {
    use instant::Instant;
    
    let batch_count = coeffs_batches.len();
    assert_eq!(batch_count, bases_batches.len(), "Number of coefficient and base batches must match");
    
    if batch_count == 0 {
        return Vec::new();
    }
    
    let msm_start = Instant::now();
    log::debug!("🚀 [BATCHED_MSM_GPU] Starting batched GPU MSM: {} batches", batch_count);
    
    // For now, we'll use parallel individual MSM calls
    // In the future, this could be optimized to use a single GPU kernel for all batches
    let results: Vec<C::Curve> = coeffs_batches
        .par_iter()
        .enumerate()
        .map(|(i, coeffs)| multiexp_on_device(coeffs, bases_batches[i]))
        .collect();
    
    let elapsed = msm_start.elapsed();
    let total_elements: usize = coeffs_batches.iter().map(|c| c.len()).sum();
    
    log::info!("⚡ [BATCHED_MSM_GPU] Batched GPU MSM completed: {} batches, {} total elements in {:?} ({:.2} elements/ms)", 
               batch_count, total_elements, elapsed, total_elements as f64 / elapsed.as_millis() as f64);
    
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