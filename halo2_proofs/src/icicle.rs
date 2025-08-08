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
use maybe_rayon::prelude::ParallelSlice;
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
    let start_time = std::time::Instant::now();
    
    // Optimization 0: Check if zero-copy conversion is possible
    let use_zero_copy = std::env::var("HALO2_ZERO_COPY_CONVERSION")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
    
    if use_zero_copy && coeffs.len() > 10000 {
        // Try zero-copy conversion for large datasets
        if let Ok(result) = try_zero_copy_scalar_conversion(coeffs) {
            let elapsed = start_time.elapsed();
            println!("      🚀 Zero-copy scalar conversion: {} elements in {:.2?} ({:.2} elements/ms)", 
                     coeffs.len(), elapsed, coeffs.len() as f64 / elapsed.as_millis().max(1) as f64);
            return result;
        }
    }
    
    // Optimization 1: Pre-allocate vector with exact capacity
    let mut result = Vec::with_capacity(coeffs.len());
    
    // Optimization 2: Use chunked processing for better cache locality
    let chunk_size = std::env::var("HALO2_CONVERSION_CHUNK_SIZE")
        .unwrap_or_else(|_| "8192".to_string())
        .parse::<usize>()
        .unwrap_or(8192);
    
    // Optimization 3: Use rayon's par_chunks for better parallelization
    let chunks: Vec<_> = coeffs.par_chunks(chunk_size)
        .map(|chunk| {
            chunk.iter().map(|coef| {
                let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
                ScalarField::from(repr)
            }).collect::<Vec<_>>()
        })
        .collect();
    
    // Optimization 4: Flatten results efficiently
    for chunk in chunks {
        result.extend(chunk);
    }
    
    let elapsed = start_time.elapsed();
    if coeffs.len() > 1000 {  // Only log for large operations
        println!("      🔄 Optimized scalar conversion: {} elements in {:.2?} ({:.2} elements/ms)", 
                 coeffs.len(), elapsed, coeffs.len() as f64 / elapsed.as_millis().max(1) as f64);
    }
    result
}

// Optimization: Zero-copy conversion when possible
fn try_zero_copy_scalar_conversion<G: PrimeField>(coeffs: &[G]) -> Result<Vec<ScalarField>, &'static str> {
    // This is a placeholder for zero-copy conversion
    // In practice, this would check if the memory layouts are compatible
    // and perform a direct memory copy instead of element-by-element conversion
    
    // For now, we'll fall back to the optimized conversion
    Err("Zero-copy not implemented for this field type")
}

// Optimization: Batch conversion for multiple operations
pub fn batch_icicle_scalars_from_c_scalars<G: PrimeField>(
    operations: &[&[G]]
) -> Vec<Vec<ScalarField>> {
    let start_time = std::time::Instant::now();
    let total_elements: usize = operations.iter().map(|op| op.len()).sum();
    
    println!("🚀 [BATCH_CONVERSION] Starting batch scalar conversion:");
    println!("   📊 Total operations: {}", operations.len());
    println!("   📊 Total elements: {}", total_elements);
    
    // Use parallel processing for the batch
    let results: Vec<Vec<ScalarField>> = operations.par_iter()
        .map(|coeffs| icicle_scalars_from_c_scalars(coeffs))
        .collect();
    
    let elapsed = start_time.elapsed();
    println!("✅ [BATCH_CONVERSION] Batch conversion completed in {:.2?}", elapsed);
    println!("   ⚡ Average: {:.2} operations/ms", 
             operations.len() as f64 / elapsed.as_millis().max(1) as f64);
    println!("   ⚡ Element throughput: {:.2} elements/ms", 
             total_elements as f64 / elapsed.as_millis().max(1) as f64);
    
    results
}

fn c_scalars_from_icicle_scalars<G: PrimeField>(scalars: &[ScalarField]) -> Vec<G> {
    let start_time = std::time::Instant::now();
    
    // Optimization 1: Pre-allocate vector with exact capacity
    let mut result = Vec::with_capacity(scalars.len());
    
    // Optimization 2: Use chunked processing for better cache locality
    let chunk_size = std::env::var("HALO2_CONVERSION_CHUNK_SIZE")
        .unwrap_or_else(|_| "8192".to_string())
        .parse::<usize>()
        .unwrap_or(8192);
    
    // Optimization 3: Use rayon's par_chunks for better parallelization
    let chunks: Vec<_> = scalars.par_chunks(chunk_size)
        .map(|chunk| {
            chunk.iter().map(|scalar| {
                let repr: G::Repr = unsafe { mem::transmute_copy(scalar) };
                G::from_repr(repr).unwrap()
            }).collect::<Vec<_>>()
        })
        .collect();
    
    // Optimization 4: Flatten results efficiently
    for chunk in chunks {
        result.extend(chunk);
    }
    
    let elapsed = start_time.elapsed();
    if scalars.len() > 1000 {  // Only log for large operations
        println!("      🔄 Optimized reverse scalar conversion: {} elements in {:.2?} ({:.2} elements/ms)", 
                 scalars.len(), elapsed, scalars.len() as f64 / elapsed.as_millis().max(1) as f64);
    }
    result
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    let start_time = std::time::Instant::now();
    
    // Optimization 1: Pre-allocate vector with exact capacity
    let mut result = Vec::with_capacity(bases.len());
    
    // Optimization 2: Use chunked processing for better cache locality
    let chunk_size = std::env::var("HALO2_CONVERSION_CHUNK_SIZE")
        .unwrap_or_else(|_| "8192".to_string())
        .parse::<usize>()
        .unwrap_or(8192);
    
    // Optimization 3: Use rayon's par_chunks for better parallelization
    let chunks: Vec<_> = bases.par_chunks(chunk_size)
        .map(|chunk| {
            chunk.iter().map(|p| {
                let coordinates = p.coordinates().unwrap();
                let x_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.x().to_repr()) };
                let y_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.y().to_repr()) };

                Affine::<CurveCfg>::from_limbs(x_repr, y_repr)
            }).collect::<Vec<_>>()
        })
        .collect();
    
    // Optimization 4: Flatten results efficiently
    for chunk in chunks {
        result.extend(chunk);
    }
    
    let elapsed = start_time.elapsed();
    if bases.len() > 1000 {  // Only log for large operations
        println!("      🔄 Optimized point conversion: {} elements in {:.2?} ({:.2} elements/ms)", 
                 bases.len(), elapsed, bases.len() as f64 / elapsed.as_millis().max(1) as f64);
    }
    result
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
    let start_time = std::time::Instant::now();
    let data_size = coeffs.len();
    
    println!("🚀 [GPU_MSM] Starting GPU MSM operation:");
    println!("   📊 Data size: {} elements", data_size);
    // Input buffer diagnostics (CPU)
    println!(
        "   🧠 Scalars buffer: len={} | type={} | location=CPU (&[..])",
        coeffs.len(),
        std::any::type_name::<C::Scalar>()
    );
    println!(
        "   🧠 Bases   buffer: len={} | type={} | location=CPU (&[..])",
        bases.len(),
        std::any::type_name::<C>()
    );

    // Optional: compute lightweight fingerprints to confirm data identity across calls
    if std::env::var("HALO2_MSM_FINGERPRINT").ok().as_deref() == Some("1") {
        use ff::PrimeField;
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let sample_stride = (data_size / 65536).max(1); // sample up to ~65k elements

        let mut scalar_hasher = DefaultHasher::new();
        let mut sampled_scalars = 0usize;
        for (idx, s) in coeffs.iter().enumerate().step_by(sample_stride) {
            let repr = s.to_repr();
            repr.as_ref().hash(&mut scalar_hasher);
            sampled_scalars += 1;
        }
        let scalar_fp = scalar_hasher.finish();

        let mut point_hasher = DefaultHasher::new();
        let mut sampled_points = 0usize;
        for (idx, p) in bases.iter().enumerate().step_by(sample_stride) {
            if let Some(coords) = p.coordinates().into() {
                coords.x().to_repr().as_ref().hash(&mut point_hasher);
                coords.y().to_repr().as_ref().hash(&mut point_hasher);
            } else {
                // Point at infinity marker
                0xFFFFu16.hash(&mut point_hasher);
            }
            sampled_points += 1;
        }
        let point_fp = point_hasher.finish();

        println!(
            "   🧪 Scalars fingerprint: 0x{:016x} | sampled={} stride={} (HALO2_MSM_FINGERPRINT=1)",
            scalar_fp, sampled_scalars, sample_stride
        );
        println!(
            "   🧪 Bases   fingerprint: 0x{:016x} | sampled={} stride={} (HALO2_MSM_FINGERPRINT=1)",
            point_fp, sampled_points, sample_stride
        );
    }
    
    // Step 1: Convert scalars to GPU format
    let scalar_start = std::time::Instant::now();
    let binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(coeffs);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let scalar_elapsed = scalar_start.elapsed();
    println!("   ✅ Step 1 - Scalar conversion: {:.2?} ({:.2} elements/ms)", 
             scalar_elapsed, data_size as f64 / scalar_elapsed.as_millis().max(1) as f64);
    println!(
        "   🔎 Scalars staging: len={} | kind=Icicle HostSlice | location=CPU (staging)",
        binding.len()
    );
    
    // Step 2: Convert points to GPU format
    let point_start = std::time::Instant::now();
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);
    let point_elapsed = point_start.elapsed();
    println!("   ✅ Step 2 - Point conversion: {:.2?} ({:.2} elements/ms)", 
             point_elapsed, data_size as f64 / point_elapsed.as_millis().max(1) as f64);
    println!(
        "   🔎 Points  staging: len={} | kind=Icicle HostSlice | location=CPU (staging)",
        binding.len()
    );
    
    // Step 3: Allocate GPU memory
    let alloc_start = std::time::Instant::now();
    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
    let cfg = msm::MSMConfig::default();
    let alloc_elapsed = alloc_start.elapsed();
    println!("   ✅ Step 3 - GPU memory allocation: {:.2?}", alloc_elapsed);
    println!(
        "   🔎 Results buffer: len={} | kind=Icicle DeviceVec<G1Projective> | location=GPU",
        1
    );
    
    // Step 4: Execute GPU MSM
    let gpu_start = std::time::Instant::now();
    msm::msm(coeffs, bases, &cfg, &mut msm_results[..]).unwrap();
    let gpu_elapsed = gpu_start.elapsed();
    println!("   ✅ Step 4 - GPU MSM computation: {:.2?} ({:.2} elements/ms)", 
             gpu_elapsed, data_size as f64 / gpu_elapsed.as_millis().max(1) as f64);
    
    // Step 5: Copy result back to CPU
    let copy_start = std::time::Instant::now();
    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();
    let copy_elapsed = copy_start.elapsed();
    println!("   ✅ Step 5 - GPU to CPU copy: {:.2?}", copy_elapsed);
    println!(
        "   🔎 Results host  : len={} | type={} | location=CPU (Vec)",
        msm_host_result.len(),
        std::any::type_name::<G1Projective>()
    );
    
    // Step 6: Convert result back to Halo2 format
    let convert_start = std::time::Instant::now();
    let msm_point = c_from_icicle_point::<C>(&msm_host_result[0]);
    let convert_elapsed = convert_start.elapsed();
    println!("   ✅ Step 6 - Result conversion: {:.2?}", convert_elapsed);
    println!(
        "   🔎 Return type   : {} | location=CPU",
        std::any::type_name::<C::Curve>()
    );
    
    let total_elapsed = start_time.elapsed();
    println!("✅ [GPU_MSM] GPU MSM completed in {:.2?}", total_elapsed);
    println!("   ⚡ Total throughput: {:.2} elements/ms", 
             data_size as f64 / total_elapsed.as_millis().max(1) as f64);
    println!("   📊 Breakdown:");
    println!("      - Data conversion: {:.1}%", 
             (scalar_elapsed + point_elapsed).as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - GPU computation: {:.1}%", 
             gpu_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - Memory operations: {:.1}%", 
             (alloc_elapsed + copy_elapsed + convert_elapsed).as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    
    msm_point
}

/// Batched MSM for multiple operations
pub fn batched_multiexp_on_device<C: CurveAffine>(
    operations: &[(&[C::Scalar], &[C])]
) -> Vec<C::Curve> {
    if operations.is_empty() {
        return vec![];
    }

    let start_time = std::time::Instant::now();
    let total_elements: usize = operations.iter().map(|(coeffs, _)| coeffs.len()).sum();
    println!("🚀 [GPU_BATCHED_MSM] Starting batched GPU MSM:");
    println!("   📊 Batch size: {} operations", operations.len());
    println!("   📊 Total elements: {} elements", total_elements);
    
    // Step 1: Collect all scalars and bases
    let collect_start = std::time::Instant::now();
    let mut all_scalars = Vec::new();
    let mut all_bases = Vec::new();
    let mut operation_sizes = Vec::new();
    
    for (coeffs, bases) in operations {
        operation_sizes.push(coeffs.len());
        all_scalars.extend_from_slice(coeffs);
        all_bases.extend_from_slice(bases);
    }
    let collect_elapsed = collect_start.elapsed();
    println!("   ✅ Step 1 - Data collection: {:.2?} ({:.2} elements/ms)", 
             collect_elapsed, total_elements as f64 / collect_elapsed.as_millis().max(1) as f64);
    
    // Step 2: Convert scalars to GPU format
    let scalar_start = std::time::Instant::now();
    let binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(&all_scalars);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let scalar_elapsed = scalar_start.elapsed();
    println!("   ✅ Step 2 - Scalar conversion: {:.2?} ({:.2} elements/ms)", 
             scalar_elapsed, total_elements as f64 / scalar_elapsed.as_millis().max(1) as f64);
    
    // Step 3: Convert points to GPU format
    let point_start = std::time::Instant::now();
    let binding = icicle_points_from_c(&all_bases);
    let bases = HostSlice::from_slice(&binding[..]);
    let point_elapsed = point_start.elapsed();
    println!("   ✅ Step 3 - Point conversion: {:.2?} ({:.2} elements/ms)", 
             point_elapsed, total_elements as f64 / point_elapsed.as_millis().max(1) as f64);
    
    // Step 4: Initialize MSM configuration
    let alloc_start = std::time::Instant::now();
    let cfg = msm::MSMConfig::default();
    let alloc_elapsed = alloc_start.elapsed();
    println!("   ✅ Step 4 - MSM configuration: {:.2?}", alloc_elapsed);
    
    // Step 5: Initialize results array
    let copy_start = std::time::Instant::now();
    let mut msm_host_results = vec![G1Projective::zero(); operations.len()];
    let copy_elapsed = copy_start.elapsed();
    println!("   ✅ Step 5 - Results array initialization: {:.2?}", copy_elapsed);
    
    // Step 6: Process each operation
    let gpu_start = std::time::Instant::now();
    let mut offset_scalars = 0;
    let mut offset_bases = 0;
    
    for (i, &size) in operation_sizes.iter().enumerate() {
        let op_start = std::time::Instant::now();
        
        // Create slices using the correct API
        let coeffs_slice = HostSlice::from_slice(&coeffs.as_slice()[offset_scalars..offset_scalars + size]);
        let bases_slice = HostSlice::from_slice(&bases.as_slice()[offset_bases..offset_bases + size]);
        
        // For single result, we need to create a new DeviceVec for each operation
        let mut single_result = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
        
        msm::msm(coeffs_slice, bases_slice, &cfg, &mut single_result[..]).unwrap();
        
        // Copy the single result to the appropriate position in the main results array
        let mut host_result = vec![G1Projective::zero(); 1];
        single_result
            .copy_to_host(HostSlice::from_mut_slice(&mut host_result[..]))
            .unwrap();
        
        // Store in the main results array
        msm_host_results[i] = host_result[0];
        
        let op_elapsed = op_start.elapsed();
        println!("      📦 Operation {}: {} elements in {:.2?} ({:.2} elements/ms)", 
                 i, size, op_elapsed, size as f64 / op_elapsed.as_millis().max(1) as f64);
        
        offset_scalars += size;
        offset_bases += size;
    }
    
    let gpu_elapsed = gpu_start.elapsed();
    println!("   ✅ Step 6 - GPU MSM computation: {:.2?} ({:.2} elements/ms)", 
             gpu_elapsed, total_elements as f64 / gpu_elapsed.as_millis().max(1) as f64);
    
    // Step 7: Convert results back to Halo2 format
    let convert_start = std::time::Instant::now();
    let results: Vec<C::Curve> = msm_host_results
        .iter()
        .map(|point| c_from_icicle_point::<C>(point))
        .collect();
    let convert_elapsed = convert_start.elapsed();
    println!("   ✅ Step 7 - Result conversion: {:.2?}", convert_elapsed);
    
    let total_elapsed = start_time.elapsed();
    println!("✅ [GPU_BATCHED_MSM] Batched GPU MSM completed in {:.2?}", total_elapsed);
    println!("   ⚡ Total throughput: {:.2} operations/ms", 
             operations.len() as f64 / total_elapsed.as_millis().max(1) as f64);
    println!("   ⚡ Element throughput: {:.2} elements/ms", 
             total_elements as f64 / total_elapsed.as_millis().max(1) as f64);
    println!("   📊 Breakdown:");
    println!("      - Data preparation: {:.1}%", 
             (collect_elapsed + scalar_elapsed + point_elapsed).as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - GPU computation: {:.1}%", 
             gpu_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - Memory operations: {:.1}%", 
             (alloc_elapsed + copy_elapsed + convert_elapsed).as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);

    results
}

pub fn fft_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    omega: Scalar, 
    log_n: u32, 
    inverse: bool
) {
    let start_time = std::time::Instant::now();
    let data_size = scalars.len();
    
    println!("🚀 [GPU_FFT] Starting GPU FFT operation:");
    println!("   📊 Data size: {} elements", data_size);
    println!("   ⚙️  Log_n: {}", log_n);
    println!("   🔄 Inverse: {}", inverse);
    
    // Step 1: Initialize NTT configuration
    let config_start = std::time::Instant::now();
    let cfg = NTTConfig::<'_, ScalarField>::default();
    let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };
    let config_elapsed = config_start.elapsed();
    println!("   ✅ Step 1 - NTT configuration: {:.2?}", config_elapsed);
    
    // Step 2: Convert omega to GPU format
    let omega_start = std::time::Instant::now();
    let omega = icicle_scalars_from_c_scalars(&[omega]);
    initialize_domain(omega[0], &cfg.ctx, true).unwrap();
    let omega_elapsed = omega_start.elapsed();
    println!("   ✅ Step 2 - Omega conversion & domain init: {:.2?}", omega_elapsed);
    
    // Step 3: Convert scalars to GPU format (optimized)
    let scalar_start = std::time::Instant::now();
    
    // Use optimized conversion with configurable chunk size
    let mut icicle_scalars: Vec<ScalarField> = icicle_scalars_from_c_scalars(scalars);
    
    // Create GPU memory interface
    let host_scalars = HostSlice::from_mut_slice(&mut icicle_scalars);
    
    let scalar_elapsed = scalar_start.elapsed();
    println!("   ✅ Step 3 - Optimized scalar conversion: {:.2?} ({:.2} elements/ms)", 
             scalar_elapsed, data_size as f64 / scalar_elapsed.as_millis().max(1) as f64);
    
    // Step 4: Execute GPU NTT
    let gpu_start = std::time::Instant::now();
    ntt_inplace::<ScalarField, ScalarField>(
        host_scalars,
        dir,
        &cfg,
    ).unwrap();
    let gpu_elapsed = gpu_start.elapsed();
    println!("   ✅ Step 4 - GPU NTT computation: {:.2?} ({:.2} elements/ms)", 
             gpu_elapsed, data_size as f64 / gpu_elapsed.as_millis().max(1) as f64);
    
    // Step 5: Convert results back to Halo2 format
    let convert_start = std::time::Instant::now();
    let c_scalars = &c_scalars_from_icicle_scalars::<G>(&mut host_scalars.as_slice())[..];
    scalars.copy_from_slice(&c_scalars);
    let convert_elapsed = convert_start.elapsed();
    println!("   ✅ Step 5 - Result conversion: {:.2?} ({:.2} elements/ms)", 
             convert_elapsed, data_size as f64 / convert_elapsed.as_millis().max(1) as f64);
    
    let total_elapsed = start_time.elapsed();
    println!("✅ [GPU_FFT] GPU FFT completed in {:.2?}", total_elapsed);
    println!("   ⚡ Total throughput: {:.2} elements/ms", 
             data_size as f64 / total_elapsed.as_millis().max(1) as f64);
    println!("   📊 Breakdown:");
    println!("      - Setup: {:.1}%", 
             (config_elapsed + omega_elapsed).as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - GPU computation: {:.1}%", 
             gpu_elapsed.as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
    println!("      - Data conversion: {:.1}%", 
             (scalar_elapsed + convert_elapsed).as_millis() as f64 / total_elapsed.as_millis() as f64 * 100.0);
}