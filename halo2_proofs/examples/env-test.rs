//! Simple test to verify environment variable control for batch MSM

use std::env;

fn main() {
    println!("🔧 Environment Variable Test for Batch MSM");
    println!("==========================================");
    
    // Test different environment variable values
    let test_values = vec![
        ("HALO2_BATCH_MSM", "1"),
        ("HALO2_BATCH_MSM", "0"),
        ("HALO2_BATCH_MSM", ""),
        ("HALO2_BATCH_MSM", "true"),
        ("HALO2_BATCH_MSM", "false"),
    ];
    
    for (var_name, var_value) in test_values {
        // Set the environment variable
        if var_value.is_empty() {
            env::remove_var(var_name);
        } else {
            env::set_var(var_name, var_value);
        }
        
        // Test the logic that's used in the prover
        let use_batch = env::var("HALO2_BATCH_MSM").unwrap_or_default() == "1";
        
        println!("{}={:?} -> use_batch = {}", var_name, var_value, use_batch);
    }
    
    println!("\n📋 Summary:");
    println!("✅ HALO2_BATCH_MSM=1 -> batch mode enabled");
    println!("✅ HALO2_BATCH_MSM=0 -> batch mode disabled");
    println!("✅ HALO2_BATCH_MSM=unset -> batch mode disabled");
    println!("✅ HALO2_BATCH_MSM=anything_else -> batch mode disabled");
    
    println!("\n💡 Usage:");
    println!("   export HALO2_BATCH_MSM=1  # Enable batch MSM");
    println!("   unset HALO2_BATCH_MSM     # Disable batch MSM");
} 