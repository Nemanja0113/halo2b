//! Example demonstrating the batch MSM functionality in Halo2
//! 
//! This example shows how to use the batch MSM feature to improve performance
//! when committing to multiple polynomials simultaneously.

use halo2_proofs::{
    arithmetic::{BatchMSMInput, best_batch_multiexp},
    poly::{
        commitment::{Blind, Params},
        kzg::commitment::ParamsKZG,
        EvaluationDomain, LagrangeCoeff, Polynomial,
    },
};
use halo2curves::bn256::{Bn256, Fr};
use ff::Field;
use rand_core::OsRng;

fn main() {
    // Set up parameters
    let k = 4;
    let params = ParamsKZG::<Bn256>::new(k);
    let domain = EvaluationDomain::new(1, k);
    
    println!("🚀 Batch MSM Example");
    println!("Domain size: {}", domain.n());
    
    // Create test polynomials
    let mut polynomials = Vec::new();
    let mut blinding_factors = Vec::new();
    
    // Generate 6 test polynomials (like in Phase 3)
    for i in 0..6 {
        let mut poly = domain.empty_lagrange();
        for (j, value) in poly.iter_mut().enumerate() {
            *value = Fr::from((i * 1000 + j) as u64);
        }
        polynomials.push(poly);
        blinding_factors.push(Blind(Fr::random(OsRng)));
    }
    
    println!("Created {} polynomials for testing", polynomials.len());
    
    // Method 1: Original parallel approach
    println!("\n📊 Method 1: Original Parallel Approach");
    let start = std::time::Instant::now();
    
    let parallel_results: Vec<_> = polynomials
        .iter()
        .zip(blinding_factors.iter())
        .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
        .collect();
    
    let parallel_time = start.elapsed();
    println!("✅ Parallel approach completed in {:?}", parallel_time);
    println!("   Results: {} commitments", parallel_results.len());
    
    // Method 2: Batch approach
    println!("\n📊 Method 2: Batch Approach");
    let start = std::time::Instant::now();
    
    let batch_results = params.commit_lagrange_batch(
        &polynomials.iter().collect::<Vec<_>>(),
        &blinding_factors,
    );
    
    let batch_time = start.elapsed();
    println!("✅ Batch approach completed in {:?}", batch_time);
    println!("   Results: {} commitments", batch_results.len());
    
    // Verify results are the same
    println!("\n🔍 Verifying Results");
    let mut all_match = true;
    for (i, (parallel, batch)) in parallel_results.iter().zip(batch_results.iter()).enumerate() {
        if parallel != batch {
            println!("❌ Mismatch at index {}: {:?} vs {:?}", i, parallel, batch);
            all_match = false;
        }
    }
    
    if all_match {
        println!("✅ All results match!");
    } else {
        println!("❌ Results don't match!");
    }
    
    // Performance comparison
    println!("\n📈 Performance Comparison");
    if batch_time < parallel_time {
        let speedup = parallel_time.as_micros() as f64 / batch_time.as_micros() as f64;
        println!("🚀 Batch approach is {:.2}x faster!", speedup);
    } else {
        let slowdown = batch_time.as_micros() as f64 / parallel_time.as_micros() as f64;
        println!("🐌 Batch approach is {:.2}x slower", slowdown);
    }
    
    println!("\n💡 Usage Instructions:");
    println!("   - Enable batch mode: cargo build --features batch");
    println!("   - Disable batch mode: cargo build --no-default-features");
    println!("   - The system automatically chooses the best approach based on the feature flag");
} 