#!/usr/bin/env python3
"""
Quick test script for sexual vs. asexual analysis
"""

import sys
import os

# Simple, hardcoded project root - clean and obvious
project_root = os.path.expanduser("~/Documents/programming/SelectionAndSexualReproduction")

# Add project root to Python path for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sexual_vs_asexual_analysis import SexualVsAsexualAnalyzer, weak_selection

def quick_test():
    """Run a quick test with just 10 runs."""
    
    # Create analyzer with fewer runs for quick testing
    analyzer = SexualVsAsexualAnalyzer(
        population_size=100,
        genome_length=20,
        mutation_rate=0.005,
        generations=100,
        n_runs=10,  # Just 10 runs for quick testing
        base_seed=42
    )
    
    print("Running quick test with 10 simulations...")
    
    # Test with weak selection only
    df = analyzer.run_comprehensive_analysis(weak_selection, "Weak Selection (Quick Test)")
    
    print(f"\n✅ Quick test completed!")
    print(f"DataFrame shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    df = quick_test() 