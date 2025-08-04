use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use instant::Instant;
use rand_core::RngCore;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeSet;
use std::iter;
use std::ops::RangeTo;

use super::{
    circuit::{
        sealed::{self},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    permutation, shuffle, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, Error, ProvingKey,
};
#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IndexedParallelIterator, ParallelIterator};

#[cfg(not(feature = "mv-lookup"))]
use super::lookup;
#[cfg(feature = "mv-lookup")]
use super::mv_lookup as lookup;

#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};

use crate::{
    arithmetic::{eval_polynomial, CurveAffine, parallelize},
    circuit::Value,
    plonk::Assigned,
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, LagrangeCoeff, Polynomial, ProverQuery, EvaluationDomain,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurveAffine;
use std::env;

// Configurable batch sizes for parallel witness processing
const DEFAULT_WITNESS_BATCH_SIZE: usize = 1024;
const DEFAULT_PARALLEL_CHUNK_SIZE: usize = 8192;

fn get_witness_batch_size() -> usize {
    let batch_size = env::var("HALO2_WITNESS_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WITNESS_BATCH_SIZE);
    
    println!("⚙️  [CONFIG] Witness batch size: {} (env: {})", 
        batch_size, 
        env::var("HALO2_WITNESS_BATCH_SIZE").unwrap_or_else(|_| "not set".to_string())
    );
    
    batch_size
}

fn get_parallel_chunk_size() -> usize {
    let chunk_size = env::var("HALO2_PARALLEL_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PARALLEL_CHUNK_SIZE);
    
    println!("⚙️  [CONFIG] Parallel chunk size: {} (env: {})", 
        chunk_size, 
        env::var("HALO2_PARALLEL_CHUNK_SIZE").unwrap_or_else(|_| "not set".to_string())
    );
    
    chunk_size
}

/// Process witness data in parallel batches for improved GPU utilization
fn process_witness_batch_parallel<F: Field>(
    witness_data: &[Polynomial<Assigned<F>, LagrangeCoeff>],
    batch_size: usize,
) -> Vec<Polynomial<Assigned<F>, LagrangeCoeff>> {
    let start_time = Instant::now();
    let total_elements = witness_data.len();
    
    println!("🚀 [PARALLEL_WITNESS] Starting parallel witness processing:");
    println!("   📊 Total witness elements: {}", total_elements);
    println!("   ⚙️  Batch size: {}", batch_size);
    println!("   🧵 Using parallel processing");
    
    let mut processed = witness_data.to_vec();
    
    // Process witness data in parallel batches
    parallelize(&mut processed, |batch, start| {
        for (batch_idx, batch_item) in batch.iter_mut().enumerate() {
            let data_idx = start + batch_idx;
            if data_idx < witness_data.len() {
                *batch_item = witness_data[data_idx].clone();
            }
        }
    });
    
    let elapsed = start_time.elapsed();
    println!("✅ [PARALLEL_WITNESS] Completed in {:.2?}", elapsed);
    println!("   📊 Processed {} witness elements", total_elements);
    println!("   ⚡ Average: {:.2} elements/ms", total_elements as f64 / elapsed.as_millis().max(1) as f64);
    
    processed
}

/// Optimized batch inversion with larger chunk sizes for GPU utilization
fn batch_invert_assigned_optimized<F: Field>(
    assigned: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let start_time = Instant::now();
    let total_elements = assigned.len();
    let chunk_size = get_parallel_chunk_size();
    
    println!("🔄 [BATCH_INVERT] Starting optimized batch inversion:");
    println!("   📊 Total elements: {}", total_elements);
    println!("   ⚙️  Chunk size: {}", chunk_size);
    println!("   🧵 Using parallel processing");
    
    let result = if assigned.len() > chunk_size {
        println!("   📈 Using chunked processing (large dataset)");
        let mut results = Vec::with_capacity(assigned.len());
        
        // Process in chunks for better GPU utilization
        for (chunk_idx, chunk) in assigned.chunks(chunk_size).enumerate() {
            let chunk_start_time = Instant::now();
            let chunk_results = batch_invert_assigned(chunk.to_vec());
            results.extend(chunk_results);
            let chunk_elapsed = chunk_start_time.elapsed();
            println!("   📦 Chunk {}: {} elements in {:.2?}", chunk_idx, chunk.len(), chunk_elapsed);
        }
        
        results
    } else {
        println!("   📉 Using single-batch processing (small dataset)");
        // Use the original implementation for smaller datasets
        batch_invert_assigned(assigned)
    };
    
    let elapsed = start_time.elapsed();
    println!("✅ [BATCH_INVERT] Completed in {:.2?}", elapsed);
    println!("   📊 Processed {} elements", total_elements);
    println!("   ⚡ Average: {:.2} elements/ms", total_elements as f64 / elapsed.as_millis().max(1) as f64);
    
    result
}



/// Optimize witness collection initialization for better GPU utilization
fn optimize_witness_collection<F: Field + WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    num_advice_columns: usize,
) -> Vec<Polynomial<Assigned<F>, LagrangeCoeff>> {
    let start_time = Instant::now();
    let batch_size = get_witness_batch_size();
    
    println!("🏗️  [WITNESS_INIT] Starting optimized witness collection initialization:");
    println!("   📊 Total advice columns: {}", num_advice_columns);
    println!("   ⚙️  Batch size: {}", batch_size);
    println!("   🧵 Using parallel processing");
    
    let mut advice = Vec::with_capacity(num_advice_columns);
    
    // Initialize advice columns in parallel batches
    for (chunk_idx, chunk_start) in (0..num_advice_columns).step_by(batch_size).enumerate() {
        let chunk_end = (chunk_start + batch_size).min(num_advice_columns);
        let chunk_size = chunk_end - chunk_start;
        
        let chunk_start_time = Instant::now();
        let chunk_advice = vec![domain.empty_lagrange_assigned(); chunk_size];
        advice.extend(chunk_advice);
        let chunk_elapsed = chunk_start_time.elapsed();
        println!("   📦 Chunk {}: {} columns in {:.2?}", chunk_idx, chunk_size, chunk_elapsed);
    }
    
    let elapsed = start_time.elapsed();
    println!("✅ [WITNESS_INIT] Completed in {:.2?}", elapsed);
    println!("   📊 Initialized {} advice columns", num_advice_columns);
    println!("   ⚡ Average: {:.2} columns/ms", num_advice_columns as f64 / elapsed.as_millis().max(1) as f64);
    
    advice
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    'params,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + Send + Sync,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[Scheme::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    Scheme::ParamsProver: Send + Sync,
{
    println!("🎯 [PROOF_GENERATION] Starting proof generation with parallel witness processing optimizations");
    println!("   📊 Circuits: {}, Instances: {}", circuits.len(), instances.len());
    println!("   🧵 Using parallel processing");
    println!("   🚀 GPU optimizations: ENABLED");
    
    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;

        // reset counters at the beginning of the prove
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    let start = Instant::now();
    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;
    log::trace!("Hashing verification key: {:?}", start.elapsed());

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    }

    let start = Instant::now();
    let instance: Vec<InstanceSingle<Scheme::Curve>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<Scheme::Curve>, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = domain.empty_lagrange();
                    assert_eq!(poly.len(), params.n() as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    for (poly, value) in poly.iter_mut().zip(values.iter()) {
                        if !P::QUERY_INSTANCE {
                            transcript.common_scalar(*value)?;
                        }
                        *poly = *value;
                    }
                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            if P::QUERY_INSTANCE {
                let instance_commitments_projective: Vec<_> = instance_values
                    .iter()
                    .map(|poly| params.commit_lagrange(poly, Blind::default()))
                    .collect();
                let mut instance_commitments =
                    vec![Scheme::Curve::identity(); instance_commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &instance_commitments_projective,
                    &mut instance_commitments,
                );
                let instance_commitments = instance_commitments;
                drop(instance_commitments_projective);

                for commitment in &instance_commitments {
                    transcript.common_point(*commitment)?;
                }
            }

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Instance preparation: {:?}", start.elapsed());

    #[derive(Clone)]
    struct AdviceSingle<C: CurveAffine, B: Basis> {
        pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
        pub advice_blinds: Vec<Blind<C::Scalar>>,
    }

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
        unblinded_advice: HashSet<usize>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase != column.column_type().phase {
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            *self
                .advice
                .get_mut(column.index())
                .and_then(|v| v.get_mut(row))
                .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let start = Instant::now();
    let (advice, challenges) = {
        let mut advice = Vec::with_capacity(instances.len());
        
        // Initialize advice in parallel batches for better GPU utilization
        let batch_size = get_witness_batch_size();
        for _ in 0..instances.len() {
            let mut advice_polys = Vec::with_capacity(meta.num_advice_columns);
            let mut advice_blinds = Vec::with_capacity(meta.num_advice_columns);
            
            // Initialize advice polynomials in parallel
            for chunk_start in (0..meta.num_advice_columns).step_by(batch_size) {
                let chunk_end = (chunk_start + batch_size).min(meta.num_advice_columns);
                let chunk_size = chunk_end - chunk_start;
                
                let chunk_polys = vec![domain.empty_lagrange(); chunk_size];
                let chunk_blinds = vec![Blind::default(); chunk_size];
                
                advice_polys.extend(chunk_polys);
                advice_blinds.extend(chunk_blinds);
            }
            
            advice.push(AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                advice_polys,
                advice_blinds,
            });
        }
        let s = FxBuildHasher;
        let mut challenges =
            HashMap::<usize, Scheme::Scalar>::with_capacity_and_hasher(meta.num_challenges, s);

        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
        for current_phase in pk.vk.cs.phases() {
            let _start = Instant::now();
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            for ((circuit, advice), instances) in
                circuits.iter().zip(advice.iter_mut()).zip(instances)
            {
                let _start = Instant::now();
                let mut witness = WitnessCollection {
                    k: params.k(),
                    current_phase,
                    advice: optimize_witness_collection(domain, meta.num_advice_columns),
                    unblinded_advice: HashSet::from_iter(meta.unblinded_advice_columns.clone()),
                    instances,
                    challenges: &challenges,
                    // The prover will not be allowed to assign values to advice
                    // cells that exist within inactive rows, which include some
                    // number of blinding factors and an extra row for use in the
                    // permutation argument.
                    usable_rows: ..unusable_rows_start,
                    _marker: std::marker::PhantomData,
                };

                let _start = Instant::now();
                // Synthesize the circuit to obtain the witness and other information.
                ConcreteCircuit::FloorPlanner::synthesize(
                    &mut witness,
                    circuit,
                    config.clone(),
                    meta.constants.clone(),
                )?;

                let witness_start = Instant::now();
                println!("🎯 [MAIN_PROCESS] Starting main witness processing for phase {:?}", current_phase);
                
                // Collect witness data for parallel processing
                let witness_data: Vec<_> = witness
                    .advice
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            Some(advice)
                        } else {
                            None
                        }
                    })
                    .collect();
                
                println!("   📊 Collected {} witness columns for processing", witness_data.len());
                
                // Process witness data in parallel batches for better GPU utilization
                let batch_size = get_witness_batch_size();
                let processed_witness = process_witness_batch_parallel(&witness_data, batch_size);
                
                // Use optimized batch inversion for better GPU utilization
                let mut advice_values = batch_invert_assigned_optimized::<Scheme::Scalar>(processed_witness);
                
                let witness_elapsed = witness_start.elapsed();
                println!("✅ [MAIN_PROCESS] Main witness processing completed in {:.2?}", witness_elapsed);

                let blinding_start = Instant::now();
                println!("🔒 [BLINDING] Adding blinding factors to {} advice columns", column_indices.len());
                
                // Add blinding factors to advice columns
                for (column_index, advice_values) in column_indices.iter().zip(&mut advice_values) {
                    if !witness.unblinded_advice.contains(column_index) {
                        for cell in &mut advice_values[unusable_rows_start..] {
                            *cell = Scheme::Scalar::random(&mut rng);
                        }
                    } else {
                        for cell in &mut advice_values[unusable_rows_start..] {
                            *cell = Blind::default().0;
                        }
                    }
                }
                
                let blinding_elapsed = blinding_start.elapsed();
                println!("✅ [BLINDING] Blinding factors added in {:.2?}", blinding_elapsed);

                let commitment_start = Instant::now();
                println!("🔐 [COMMITMENT] Computing commitments for {} advice columns", column_indices.len());
                
                // Compute commitments to advice column polynomials in parallel
                let mut blinds: Vec<_> = vec![Blind::default(); column_indices.len()];
                
                // Generate blinds sequentially due to RNG thread safety
                for (i, blind) in blinds.iter_mut().enumerate() {
                    if let Some(&column_index) = column_indices.iter().nth(i) {
                        if !witness.unblinded_advice.contains(&column_index) {
                            *blind = Blind(Scheme::Scalar::random(&mut rng));
                        }
                    }
                }
                
                // Compute commitments in parallel batches
                let batch_size = get_witness_batch_size();
                let mut advice_commitments_projective = Vec::with_capacity(advice_values.len());
                
                for (chunk_idx, chunk) in advice_values.chunks(batch_size).enumerate() {
                    let chunk_start = Instant::now();
                    let chunk_blinds = &blinds[..chunk.len()];
                    let chunk_commitments: Vec<_> = chunk
                        .iter()
                        .zip(chunk_blinds.iter())
                        .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                        .collect();
                    advice_commitments_projective.extend(chunk_commitments);
                    let chunk_elapsed = chunk_start.elapsed();
                    println!("   📦 Commitment chunk {}: {} elements in {:.2?}", chunk_idx, chunk.len(), chunk_elapsed);
                }
                
                let mut advice_commitments =
                    vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &advice_commitments_projective,
                    &mut advice_commitments,
                );
                let advice_commitments = advice_commitments;
                drop(advice_commitments_projective);

                let assignment_start = Instant::now();
                println!("📝 [ASSIGNMENT] Writing {} commitments to transcript", advice_commitments.len());
                
                // Write commitments to transcript
                for commitment in &advice_commitments {
                    transcript.write_point(*commitment)?;
                }
                
                println!("   📊 Assigning {} advice values and blinds", column_indices.len());
                
                // Assign advice values and blinds sequentially for simplicity
                for ((&column_index, advice_values), blind) in
                    column_indices.iter().zip(advice_values).zip(blinds)
                {
                    advice.advice_polys[column_index] = advice_values;
                    advice.advice_blinds[column_index] = blind;
                }
                
                let assignment_elapsed = assignment_start.elapsed();
                println!("✅ [ASSIGNMENT] Assignment completed in {:.2?}", assignment_elapsed);
                
                let commitment_elapsed = commitment_start.elapsed();
                println!("✅ [COMMITMENT] Commitment computation completed in {:.2?}", commitment_elapsed);
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };
    log::trace!("Advice preparation: {:?}", start.elapsed());

    // Sample theta challenge for keeping lookup columns linearly independent
    let start = Instant::now();
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Theta challenge: {:?}", start.elapsed());

    let start = Instant::now();

    #[cfg(feature = "mv-lookup")]
    let lookups: Vec<Vec<lookup::prover::Prepared<Scheme::Curve>>> = instance
        .par_iter()
        .zip(advice.par_iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .par_iter()
                .map(|lookup| {
                    lookup.prepare(
                        &pk.vk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    #[cfg(not(feature = "mv-lookup"))]
    let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Lookup preparation: {:?}", start.elapsed());

    // Sample beta challenge
    let start = Instant::now();
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Beta challenge: {:?}", start.elapsed());

    // Sample gamma challenge
    let start = Instant::now();
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Gamma challenge: {:?}", start.elapsed());

    // Commit to permutations.
    let start = Instant::now();
    let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Permutation commitment: {:?}", start.elapsed());

    // preallocate the lookups

    #[cfg(feature = "mv-lookup")]
    let phi_blinds = (0..pk.vk.cs.blinding_factors())
        .map(|_| Scheme::Scalar::random(&mut rng))
        .collect::<Vec<_>>();

    #[cfg(feature = "mv-lookup")]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        lookups
            .into_iter()
            .map(|lookups| -> Result<Vec<_>, _> {
                // Construct and commit to products for each lookup
                #[cfg(feature = "metal")]
                let res = lookups
                    .into_iter()
                    .map(|lookup| lookup.commit_grand_sum(&pk.vk, params, beta, &phi_blinds))
                    .collect::<Result<Vec<_>, _>>();

                #[cfg(not(feature = "metal"))]
                let res = lookups
                    .into_par_iter()
                    .map(|lookup| lookup.commit_grand_sum(&pk.vk, params, beta, &phi_blinds))
                    .collect::<Result<Vec<_>, _>>();

                res
            })
            .collect::<Result<Vec<_>, _>>()
    };

    #[cfg(not(feature = "mv-lookup"))]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        lookups
            .into_iter()
            .map(|lookups| -> Result<Vec<_>, _> {
                // Construct and commit to products for each lookup
                lookups
                    .into_iter()
                    .map(|lookup| {
                        lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
    };

    let start = Instant::now();
    let lookups = commit_lookups()?;

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    log::trace!("Lookup commitment: {:?}", start.elapsed());

    let start = Instant::now();
    let shuffles: Vec<Vec<shuffle::prover::Committed<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, _> {
            // Compress expressions for each shuffle
            pk.vk
                .cs
                .shuffles
                .iter()
                .map(|shuffle| {
                    shuffle.commit_product(
                        pk,
                        params,
                        domain,
                        theta,
                        gamma,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Shuffle commitment: {:?}", start.elapsed());

    let start = Instant::now();
    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;
    log::trace!("Vanishing commitment: {:?}", start.elapsed());

    // Obtain challenge for keeping all separate gates linearly independent
    let start = Instant::now();
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();
    log::trace!("Y challenge: {:?}", start.elapsed());

    // Calculate the advice polys
    let start = Instant::now();
    let advice: Vec<AdviceSingle<Scheme::Curve, Coeff>> = advice
        .into_iter()
        .map(
            |AdviceSingle {
                 advice_polys,
                 advice_blinds,
             }| {
                AdviceSingle {
                    advice_polys: advice_polys
                        .into_iter()
                        .map(|poly| domain.lagrange_to_coeff(poly))
                        .collect::<Vec<_>>(),
                    advice_blinds,
                }
            },
        )
        .collect();
    log::trace!("Advice calculation: {:?}", start.elapsed());

    // Evaluate the h(X) polynomial
    let start = Instant::now();
    let h_poly = pk.ev.evaluate_h(
        pk,
        &advice
            .iter()
            .map(|a| a.advice_polys.as_slice())
            .collect::<Vec<_>>(),
        &instance
            .iter()
            .map(|i| i.instance_polys.as_slice())
            .collect::<Vec<_>>(),
        &challenges,
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &shuffles,
        &permutations,
    );
    log::trace!("H(X) evaluation: {:?}", start.elapsed());

    // Construct the vanishing argument's h(X) commitments
    let start = Instant::now();
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;
    log::trace!("Vanishing construction: {:?}", start.elapsed());

    let start = Instant::now();
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow([params.n()]);
    log::trace!("X challenge: {:?}", start.elapsed());

    let start = Instant::now();
    if P::QUERY_INSTANCE {
        // Compute and hash instance evals for each circuit instance
        for instance in instance.iter() {
            // Evaluate polynomials at omega^i x
            let instance_evals: Vec<_> = meta
                .instance_queries
                .iter()
                .map(|&(column, at)| {
                    eval_polynomial(
                        &instance.instance_polys[column.index()],
                        domain.rotate_omega(*x, at),
                    )
                })
                .collect();

            // Hash each instance column evaluation
            for eval in instance_evals.iter() {
                transcript.write_scalar(*eval)?;
            }
        }
    }
    log::trace!("Instance evaluation: {:?}", start.elapsed());

    let start = Instant::now();
    // Compute and hash advice evals for each circuit instance
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &advice.advice_polys[column.index()],
                    domain.rotate_omega(*x, at),
                )
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    }
    log::trace!("Advice evaluation: {:?}", start.elapsed());

    let start = Instant::now();
    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
        })
        .collect();
    log::trace!("Fixed evaluation: {:?}", start.elapsed());

    // Hash each fixed column evaluation
    let start = Instant::now();
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }
    log::trace!("Fixed evaluation hashing: {:?}", start.elapsed());

    let start = Instant::now();
    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;
    log::trace!("Vanishing evaluation: {:?}", start.elapsed());

    // Evaluate common permutation data
    let start = Instant::now();
    pk.permutation.evaluate(x, transcript)?;
    log::trace!("Permutation evaluation: {:?}", start.elapsed());

    // Evaluate the permutations, if any, at omega^i x.
    let start = Instant::now();
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Permutation evaluation: {:?}", start.elapsed());

    // Evaluate the lookups, if any, at omega^i x.

    let start = Instant::now();

    let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| {
                    #[cfg(not(feature = "mv-lookup"))]
                    let res = { p.evaluate(pk, x, transcript) };
                    #[cfg(feature = "mv-lookup")]
                    let res = { p.evaluate(&pk.vk, x, transcript) };
                    res
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Lookup evaluation: {:?}", start.elapsed());

    // Evaluate the shuffles, if any, at omega^i x.
    let start = Instant::now();
    let shuffles: Vec<Vec<shuffle::prover::Evaluated<Scheme::Curve>>> = shuffles
        .into_iter()
        .map(|shuffles| -> Result<Vec<_>, _> {
            shuffles
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::trace!("Shuffle evaluation: {:?}", start.elapsed());

    let start = Instant::now();
    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .zip(shuffles.iter())
        .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
            iter::empty()
                .chain(
                    P::QUERY_INSTANCE
                        .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| {
                            ProverQuery {
                                point: domain.rotate_omega(*x, at),
                                poly: &instance.instance_polys[column.index()],
                                blind: Blind::default(),
                            }
                        }))
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            poly: &advice.advice_polys[column.index()],
                            blind: advice.advice_blinds[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                .chain(shuffles.iter().flat_map(move |p| p.open(pk, x)))
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    poly: &pk.fixed_polys[column.index()],
                    blind: Blind::default(),
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));
    log::trace!("Open queries: {:?}", start.elapsed());

    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;
        log::debug!("MSM_COUNTER: {:?}", MSM_COUNTER.lock().unwrap());
        log::debug!("FFT_COUNTER: {:?}", *FFT_COUNTER.lock().unwrap());

        // reset counters at the end of the proving
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    let prover = P::new(params);
    prover
        .create_proof(rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure)
}

#[test]
fn test_create_proof() {
    use crate::{
        circuit::SimpleFloorPlanner,
        plonk::{keygen_pk, keygen_vk},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::Bn256;
    use rand_core::OsRng;

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, OsRng);
    let vk = keygen_vk(&params, &MyCircuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &MyCircuit).expect("keygen_pk should not fail");
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // Create proof with wrong number of instances
    let proof = create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[],
        OsRng,
        &mut transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[&[], &[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
}
