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

// Add parallel processing support for synthesis optimization
use maybe_rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    circuit::Value,
    plonk::Assigned,
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurveAffine;

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
    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;

        // reset counters at the beginning of the prove
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    // Reset MSM statistics at the beginning of proof generation
    crate::arithmetic::reset_msm_stats();
    
    // Reset FFT statistics at the beginning of proof generation
    crate::arithmetic::reset_fft_stats();

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    // Phase 1: Initialization and Validation
    let phase1_start = Instant::now();
    
    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;
    
    log::info!("🔄 [PHASE 1] Initialization and Validation: {:?}", phase1_start.elapsed());

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

    // Phase 2: Instance Preparation
    let phase2_start = Instant::now();
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
    log::info!("🔄 [PHASE 2] Instance Preparation: {:?}", phase2_start.elapsed());

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

    // Phase 3: Witness Collection and Advice Preparation
    let phase3_start = Instant::now();
    let (advice, challenges) = {
        let mut advice = vec![
            AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
                advice_blinds: vec![Blind::default(); meta.num_advice_columns],
            };
            instances.len()
        ];

        let s = FxBuildHasher;
        let mut challenges =
            HashMap::<usize, Scheme::Scalar>::with_capacity_and_hasher(meta.num_challenges, s);

        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
        for current_phase in pk.vk.cs.phases() {
            let phase_sub_start = Instant::now();
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

            // Check if we should use parallel synthesis for better performance
            let use_parallel_synthesis = std::env::var("HALO2_PARALLEL_SYNTHESIS").unwrap_or_default() == "1";
            
            // Pre-allocate witness collections to reduce memory allocations
            let mut witness_pool: Vec<WitnessCollection<Scheme::Scalar>> = Vec::with_capacity(circuits.len());
            for _ in 0..circuits.len() {
                witness_pool.push(WitnessCollection {
                    k: params.k(),
                    current_phase,
                    advice: vec![domain.empty_lagrange_assigned(); meta.num_advice_columns],
                    unblinded_advice: HashSet::from_iter(meta.unblinded_advice_columns.clone()),
                    instances: Vec::new(), // Will be set per circuit
                    challenges: &challenges,
                    usable_rows: ..unusable_rows_start,
                    _marker: std::marker::PhantomData,
                });
            }
            
            // Check if we should use synthesis caching for repeated circuits
            let use_synthesis_cache = std::env::var("HALO2_SYNTHESIS_CACHE").unwrap_or_default() == "1";
            let mut synthesis_cache: HashMap<u64, WitnessCollection<Scheme::Scalar>> = HashMap::default();
            
            if use_parallel_synthesis && circuits.len() > 1 {
                // Parallel synthesis for multiple circuits
                log::info!("🚀 [SYNTHESIS] Using parallel synthesis for {} circuits", circuits.len());
                
                let synthesis_results: Vec<_> = circuits
                    .par_iter()
                    .zip(advice.par_iter_mut())
                    .zip(instances)
                    .enumerate()
                    .map(|(idx, ((circuit, advice), instances))| {
                        let synthesis_start = Instant::now();
                        
                        // Reuse pre-allocated witness collection from pool
                        let mut witness = std::mem::take(&mut witness_pool[idx]);
                        witness.instances = instances;
                        
                        // Reset advice columns without reallocating
                        for advice_col in &mut witness.advice {
                            advice_col.clear();
                            advice_col.extend(domain.empty_lagrange_assigned());
                        }
                        
                        // Check synthesis cache if enabled
                        let circuit_hash = if use_synthesis_cache {
                            // Simple hash based on circuit instance data
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            use std::hash::{Hash, Hasher};
                            instances.hash(&mut hasher);
                            hasher.finish()
                        } else {
                            0
                        };
                        
                        if use_synthesis_cache && synthesis_cache.contains_key(&circuit_hash) {
                            log::debug!("    Circuit {}: Using cached synthesis result", idx);
                            witness = synthesis_cache.get(&circuit_hash).unwrap().clone();
                        } else {
                            // Synthesize the circuit
                            let result = ConcreteCircuit::FloorPlanner::synthesize(
                                &mut witness,
                                circuit,
                                config.clone(),
                                meta.constants.clone(),
                            );
                            
                            let synthesis_elapsed = synthesis_start.elapsed();
                            log::debug!("    Circuit synthesis: {:?}", synthesis_elapsed);
                            
                            // Cache the result if caching is enabled
                            if use_synthesis_cache && result.is_ok() {
                                synthesis_cache.insert(circuit_hash, witness.clone());
                            }
                            
                            // Return witness for further processing
                            match result {
                                Ok(()) => Ok(witness),
                                Err(e) => Err(e),
                            }
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                
                // Process synthesis results sequentially for post-processing
                for (idx, ((circuit, advice), instances)) in 
                    circuits.iter().zip(advice.iter_mut()).zip(instances).enumerate() 
                {
                    let witness = synthesis_results[idx].clone();
                    
                    // Continue with batch invert, blinding, and commitments...
                    let batch_invert_start = Instant::now();
                    let mut advice_values = batch_invert_assigned::<Scheme::Scalar>(
                        witness
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
                            .collect(),
                    );
                    log::debug!("    Batch invert: {:?}", batch_invert_start.elapsed());

                    let blinding_start = Instant::now();
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
                    log::debug!("    Blinding factors: {:?}", blinding_start.elapsed());

                    let commitment_start = Instant::now();
                    // Compute commitments to advice column polynomials
                    let blinds: Vec<_> = column_indices
                        .iter()
                        .map(|i| {
                            if witness.unblinded_advice.contains(i) {
                                Blind::default()
                            } else {
                                Blind(Scheme::Scalar::random(&mut rng))
                            }
                        })
                        .collect();

                    let advice_commitments_projective: Vec<_> = {
                        // Check environment variable for batch mode
                        let use_batch = std::env::var("HALO2_BATCH_MSM").unwrap_or_default() == "1";

                        let batch_start = Instant::now();
                        
                        if use_batch {
                            // Use batch MSM for better performance
                            let polynomials: Vec<_> = advice_values.iter().collect();
                            let result = params.commit_lagrange_batch(&polynomials, &blinds);
                            log::info!("BATCH TOOK :::::: {:?}", batch_start.elapsed());
                            result
                        } else {
                            // Use original parallel approach
                            let result = advice_values
                                .iter()
                                .zip(blinds.iter())
                                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                                .collect();
                            log::info!("GPU TOOK ::::::: {:?}", batch_start.elapsed());
                            result
                        }
                    };

                    let mut advice_commitments =
                        vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
                    <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                        &advice_commitments_projective,
                        &mut advice_commitments,
                    );
                    *advice = advice_commitments;
                    drop(advice_commitments_projective);

                    log::debug!("    Advice commitments: {:?}", commitment_start.elapsed());
                }
            } else {
                // Sequential synthesis (original approach)
                for ((circuit, advice), instances) in
                    circuits.iter().zip(advice.iter_mut()).zip(instances)
                {
                    let circuit_start = Instant::now();
                    let mut witness = WitnessCollection {
                        k: params.k(),
                        current_phase,
                        advice: vec![domain.empty_lagrange_assigned(); meta.num_advice_columns],
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

                    let synthesis_start = Instant::now();
                    // Synthesize the circuit to obtain the witness and other information.
                    ConcreteCircuit::FloorPlanner::synthesize(
                        &mut witness,
                        circuit,
                        config.clone(),
                        meta.constants.clone(),
                    )?;
                    log::debug!("    Circuit synthesis: {:?}", synthesis_start.elapsed());

                let batch_invert_start = Instant::now();
                let mut advice_values = batch_invert_assigned::<Scheme::Scalar>(
                    witness
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
                        .collect(),
                );
                log::debug!("    Batch invert: {:?}", batch_invert_start.elapsed());

                let blinding_start = Instant::now();
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
                log::debug!("    Blinding factors: {:?}", blinding_start.elapsed());

                let commitment_start = Instant::now();
                // Compute commitments to advice column polynomials
                let blinds: Vec<_> = column_indices
                    .iter()
                    .map(|i| {
                        if witness.unblinded_advice.contains(i) {
                            Blind::default()
                        } else {
                            Blind(Scheme::Scalar::random(&mut rng))
                        }
                    })
                    .collect();

                let advice_commitments_projective: Vec<_> = {
                    // Check environment variable for batch mode
                    let use_batch = std::env::var("HALO2_BATCH_MSM").unwrap_or_default() == "1";

                    let batch_start = Instant::now();
                    
                    if use_batch {
                        // Use batch MSM for better performance
                        let polynomials: Vec<_> = advice_values.iter().collect();
                        let result = params.commit_lagrange_batch(&polynomials, &blinds);
                        log::info!("BATCH TOOK :::::: {:?}", batch_start.elapsed());
                        result
                    } else {
                        // Use original parallel approach
                        let result = advice_values
                            .iter()
                            .zip(blinds.iter())
                            .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                            .collect();
                        log::info!("GPU TOOK ::::::: {:?}", batch_start.elapsed());
                        result
                    }
                };

                let mut advice_commitments =
                    vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &advice_commitments_projective,
                    &mut advice_commitments,
                );
                let advice_commitments = advice_commitments;
                drop(advice_commitments_projective);

                log::debug!("    Advice commitments: {:?}", commitment_start.elapsed());

                let transcript_start = Instant::now();
                for commitment in &advice_commitments {
                    transcript.write_point(*commitment)?;
                }
                for ((column_index, advice_values), blind) in
                    column_indices.iter().zip(advice_values).zip(blinds)
                {
                    advice.advice_polys[*column_index] = advice_values;
                    advice.advice_blinds[*column_index] = blind;
                }
                log::debug!("    Transcript updates: {:?}", transcript_start.elapsed());
                log::debug!("    Total circuit processing: {:?}", circuit_start.elapsed());
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
            log::debug!("  Phase {:?} completed: {:?}", current_phase, phase_sub_start.elapsed());
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };
    log::info!("🔄 [PHASE 3] Witness Collection and Advice Preparation: {:?}", phase3_start.elapsed());

    // Phase 4: Lookup Preparation
    let phase4_start = Instant::now();
    
    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    let start = Instant::now();

    #[cfg(feature = "mv-lookup")]
    log::info!("LOOKUP PREPARATION - 1:");
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
    log::info!("LOOKUP PREPARATION - 1 ::: end");

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
    log::info!("🔄 [PHASE 4] Lookup Preparation: {:?}", phase4_start.elapsed());

    // Phase 5: Permutation Commitment
    let phase5_start = Instant::now();
    
    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    
    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();
    
    // Commit to permutations.
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
    log::info!("🔄 [PHASE 5] Permutation Commitment: {:?}", phase5_start.elapsed());

    // preallocate the lookups

    log::info!("look 1:");
    #[cfg(feature = "mv-lookup")]
    let phi_blinds = (0..pk.vk.cs.blinding_factors())
        .map(|_| Scheme::Scalar::random(&mut rng))
        .collect::<Vec<_>>();

    log::info!("look 2:");
    #[cfg(feature = "mv-lookup")]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        lookups
            .into_iter()
            .map(|lookups| -> Result<Vec<_>, _> {
                // Construct and commit to products for each lookup
                log::info!("look 2.1:");
                #[cfg(feature = "metal")]
                let res = lookups
                    .into_iter()
                    .map(|lookup| lookup.commit_grand_sum(&pk.vk, params, beta, &phi_blinds))
                    .collect::<Result<Vec<_>, _>>();

                log::info!("look 2.2:");
                #[cfg(not(feature = "metal"))]
                let res = lookups
                    .into_par_iter()
                    .map(|lookup| lookup.commit_grand_sum(&pk.vk, params, beta, &phi_blinds))
                    .collect::<Result<Vec<_>, _>>();

                res
            })
            .collect::<Result<Vec<_>, _>>()
    };
    log::info!("look 3:");

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
    log::info!("look 4:");

    // Phase 6: Lookup Product Commitments
    let phase6_start = Instant::now();
    let lookups = commit_lookups()?;

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                log::info!("look 5:");
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    log::info!("🔄 [PHASE 6] Lookup Product Commitments: {:?}", phase6_start.elapsed());

    // Phase 7: Shuffle Commitments
    let phase7_start = Instant::now();
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
    log::info!("🔄 [PHASE 7] Shuffle Commitments: {:?}", phase7_start.elapsed());

    // Phase 8: Vanishing Argument
    let phase8_start = Instant::now();
    
    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    // Calculate the advice polys
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

    // Evaluate the h(X) polynomial
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

    // Construct the vanishing argument's h(X) commitments
    let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;
    
    log::info!("🔄 [PHASE 8] Vanishing Argument: {:?}", phase8_start.elapsed());

    // Phase 9: Challenge Generation and Evaluation
    let phase9_start = Instant::now();
    
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow([params.n()]);

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

    let start = Instant::now();
    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
        })
        .collect();

    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }

    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the lookups, if any, at omega^i x.

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

    // Evaluate the shuffles, if any, at omega^i x.
    let shuffles: Vec<Vec<shuffle::prover::Evaluated<Scheme::Curve>>> = shuffles
        .into_iter()
        .map(|shuffles| -> Result<Vec<_>, _> {
            shuffles
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

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
    
    log::info!("🔄 [PHASE 9] Challenge Generation and Evaluation: {:?}", phase9_start.elapsed());

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

    // Phase 10: Final Multi-Open Proof
    let phase10_start = Instant::now();
    
    let prover = P::new(params);
    let result = prover
        .create_proof(rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure);
    
    log::info!("🔄 [PHASE 10] Final Multi-Open Proof: {:?}", phase10_start.elapsed());
    
    // Total proof generation time
    let total_start = phase1_start;
    log::info!("🚀 [TOTAL] Complete Proof Generation: {:?}", total_start.elapsed());
    
    // Print MSM statistics
    let (total_msm_count, total_msm_time, gpu_count, cpu_count, metal_count) = crate::arithmetic::get_msm_stats();
    log::info!("📊 [MSM_STATS] Total MSM operations: {} (GPU: {}, CPU: {}, Metal: {})", 
               total_msm_count, gpu_count, cpu_count, metal_count);
    log::info!("📊 [MSM_STATS] Total MSM time: {:?} ({:.2}% of total)", 
               total_msm_time, (total_msm_time.as_millis() as f64 / total_start.elapsed().as_millis() as f64) * 100.0);
    if total_msm_count > 0 {
        log::info!("📊 [MSM_STATS] Average MSM time: {:?}", total_msm_time / total_msm_count as u32);
    }
    
    // Print FFT statistics
    let (total_fft_count, total_fft_time, fft_gpu_count, fft_cpu_count) = crate::arithmetic::get_fft_stats();
    log::info!("📊 [FFT_STATS] Total FFT operations: {} (GPU: {}, CPU: {})", 
               total_fft_count, fft_gpu_count, fft_cpu_count);
    log::info!("📊 [FFT_STATS] Total FFT time: {:?} ({:.2}% of total)", 
               total_fft_time, (total_fft_time.as_millis() as f64 / total_start.elapsed().as_millis() as f64) * 100.0);
    if total_fft_count > 0 {
        log::info!("📊 [FFT_STATS] Average FFT time: {:?}", total_fft_time / total_fft_count as u32);
    }
    
    result
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
