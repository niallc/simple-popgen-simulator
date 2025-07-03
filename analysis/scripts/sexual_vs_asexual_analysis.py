#!/usr/bin/env python3
"""
Sexual vs. Asexual Reproduction Analysis
========================================

This script runs multiple simulations comparing sexual vs. asexual reproduction
and collects comprehensive statistics on fitness evolution across decile points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
from typing import Dict, List, Tuple, Any

# Robust import setup
try:
    from import_utils import quick_setup
    Simulation = quick_setup()
except ImportError:
    try:
        from src.simulation import Simulation
        print("✅ Direct import successful")
    except ImportError:
        import sys
        import os
        current_dir = os.getcwd()
        
        # Try different possible locations for src/
        if os.path.exists('src'):
            sys.path.insert(0, current_dir)
        elif os.path.exists('../src'):
            sys.path.insert(0, os.path.dirname(current_dir))
        elif os.path.exists('../../src'):
            sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
        else:
            raise ImportError("Cannot locate src/ directory")
        
        from src.simulation import Simulation
        print("✅ Manual path setup successful")

# Define fitness functions
def neutral_fitness(population):
    """True neutral evolution: all individuals have equal fitness"""
    return np.ones(len(population.genomes))

def weak_selection(population):
    """Weak selection: fitness = 1 + s * (number of 1s)"""
    s = 0.05  # weak selection coefficient
    base_fitness = np.array([np.sum(g.sequence) for g in population.genomes])
    return 1.0 + s * base_fitness

def additive_fitness(population):
    """Additive selection: fitness = sum of 1s in genome"""
    return np.array([np.sum(g.sequence) for g in population.genomes])

class SexualVsAsexualAnalyzer:
    """Analyzer for comparing sexual vs. asexual reproduction across multiple runs."""
    
    def __init__(self, 
                 population_size: int = 100,
                 genome_length: int = 20,
                 mutation_rate: float = 0.005,
                 generations: int = 100,
                 n_runs: int = 100,
                 base_seed: int = 42):
        """
        Initialize the analyzer with simulation parameters.
        
        Args:
            population_size: Number of individuals in population
            genome_length: Length of binary genome
            mutation_rate: Per-locus mutation rate
            generations: Number of generations to simulate
            n_runs: Number of independent simulation runs
            base_seed: Base seed for reproducible random number generation
        """
        self.params = {
            'population_size': population_size,
            'genome_length': genome_length,
            'mutation_rate': mutation_rate,
            'generations': generations,
            'random_seed': base_seed
        }
        self.n_runs = n_runs
        self.base_seed = base_seed
        
        # Calculate decile points (0th, 10th, 20th, ..., 90th, 100th percentile)
        self.decile_points = np.linspace(0, generations, 11, dtype=int)
        self.decile_labels = [f"{i*10}%" for i in range(11)]
        
        # Storage for results
        self.results = {
            'sexual': [],
            'asexual': []
        }
        
    def run_single_comparison(self, run_id: int, fitness_function) -> Tuple[Dict, Dict]:
        """
        Run a single comparison between sexual and asexual reproduction.
        
        Args:
            run_id: Run identifier for seed generation
            fitness_function: Fitness function to use
            
        Returns:
            Tuple of (sexual_results, asexual_results)
        """
        # Use different seeds for each run but keep them close for fair comparison
        sexual_seed = self.base_seed + run_id * 2
        asexual_seed = self.base_seed + run_id * 2 + 1
        
        # Create parameter dictionaries with specific seeds
        sexual_params = self.params.copy()
        sexual_params['random_seed'] = sexual_seed
        
        asexual_params = self.params.copy()
        asexual_params['random_seed'] = asexual_seed
        
        # Run sexual simulation
        sexual_sim = Simulation(
            **sexual_params, 
            mode='sexual', 
            fitness_function=fitness_function
        )
        sexual_sim.run()
        sexual_results = sexual_sim.get_results()
        
        # Run asexual simulation
        asexual_sim = Simulation(
            **asexual_params, 
            mode='asexual', 
            fitness_function=fitness_function
        )
        asexual_sim.run()
        asexual_results = asexual_sim.get_results()
        
        return sexual_results, asexual_results
    
    def normalize_fitness(self, fitness_series: pd.Series) -> pd.Series:
        """Normalize fitness to start at 1.0 for fair comparison."""
        initial_fitness = fitness_series.iloc[0]
        return fitness_series / initial_fitness
    
    def extract_decile_fitness(self, results: Dict) -> List[float]:
        """Extract fitness values at decile points and normalize."""
        fitness_series = results['mean_fitness']
        normalized_fitness = self.normalize_fitness(fitness_series)
        
        decile_fitness = []
        for point in self.decile_points:
            if point < len(normalized_fitness):
                decile_fitness.append(normalized_fitness.iloc[point])
            else:
                decile_fitness.append(normalized_fitness.iloc[-1])
        
        return decile_fitness
    
    def run_comprehensive_analysis(self, fitness_function, fitness_name: str = "Custom"):
        """
        Run comprehensive analysis comparing sexual vs. asexual reproduction.
        
        Args:
            fitness_function: Fitness function to use
            fitness_name: Name of the fitness regime for reporting
        """
        print(f"\n{'='*80}")
        print(f"SEXUAL VS. ASEXUAL REPRODUCTION ANALYSIS")
        print(f"Fitness Regime: {fitness_name}")
        print(f"{'='*80}")
        
        print(f"\nConfiguration:")
        for key, value in self.params.items():
            print(f"  {key}: {value}")
        print(f"  n_runs: {self.n_runs}")
        print(f"  decile_points: {self.decile_points}")
        
        print(f"\nRunning {self.n_runs} simulations...")
        start_time = time.time()
        
        # Run all simulations
        sexual_fitness_data = []
        asexual_fitness_data = []
        
        for run_id in range(self.n_runs):
            if (run_id + 1) % 20 == 0:
                print(f"  Completed {run_id + 1}/{self.n_runs} runs...")
            
            try:
                sexual_results, asexual_results = self.run_single_comparison(run_id, fitness_function)
                
                # Extract decile fitness values
                sexual_deciles = self.extract_decile_fitness(sexual_results)
                asexual_deciles = self.extract_decile_fitness(asexual_results)
                
                sexual_fitness_data.append(sexual_deciles)
                asexual_fitness_data.append(asexual_deciles)
                
            except Exception as e:
                print(f"  Warning: Run {run_id} failed: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"✅ Completed {len(sexual_fitness_data)} successful runs in {elapsed_time:.1f} seconds")
        
        # Convert to numpy arrays for analysis
        sexual_array = np.array(sexual_fitness_data)
        asexual_array = np.array(asexual_fitness_data)
        
        # Calculate summary statistics
        self.calculate_and_display_statistics(sexual_array, asexual_array, fitness_name)
        
        # Create comprehensive DataFrame
        df = self.create_analysis_dataframe(sexual_array, asexual_array, fitness_name)
        
        return df
    
    def calculate_and_display_statistics(self, sexual_array: np.ndarray, asexual_array: np.ndarray, fitness_name: str):
        """Calculate and display comprehensive statistics."""
        
        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        # Calculate statistics for each decile
        sexual_mean = np.mean(sexual_array, axis=0)
        sexual_std = np.std(sexual_array, axis=0)
        sexual_median = np.median(sexual_array, axis=0)
        sexual_q25 = np.percentile(sexual_array, 25, axis=0)
        sexual_q75 = np.percentile(sexual_array, 75, axis=0)
        
        asexual_mean = np.mean(asexual_array, axis=0)
        asexual_std = np.std(asexual_array, axis=0)
        asexual_median = np.median(asexual_array, axis=0)
        asexual_q25 = np.percentile(asexual_array, 25, axis=0)
        asexual_q75 = np.percentile(asexual_array, 75, axis=0)
        
        # Calculate sexual advantage
        sexual_advantage = sexual_mean - asexual_mean
        sexual_advantage_pct = (sexual_advantage / asexual_mean) * 100
        
        # Statistical significance testing
        p_values = []
        for i in range(len(self.decile_points)):
            if len(sexual_array) > 1 and len(asexual_array) > 1:
                t_stat, p_val = stats.ttest_ind(sexual_array[:, i], asexual_array[:, i])
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        
        # Display results in table format
        print(f"\n{'Decile':<8} {'Sexual':<15} {'Asexual':<15} {'Advantage':<15} {'P-value':<10}")
        print(f"{'Mean±Std':<8} {'Mean±Std':<15} {'Mean±Std':<15} {'Mean±%':<15} {'':<10}")
        print("-" * 80)
        
        for i, (point, label) in enumerate(zip(self.decile_points, self.decile_labels)):
            sexual_str = f"{sexual_mean[i]:.3f}±{sexual_std[i]:.3f}"
            asexual_str = f"{asexual_mean[i]:.3f}±{asexual_std[i]:.3f}"
            advantage_str = f"{sexual_advantage[i]:+.3f} ({sexual_advantage_pct[i]:+.1f}%)"
            p_str = f"{p_values[i]:.3f}" if p_values[i] < 0.001 else f"{p_values[i]:.3f}"
            
            print(f"{label:<8} {sexual_str:<15} {asexual_str:<15} {advantage_str:<15} {p_str:<10}")
        
        # Overall summary
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        
        final_sexual_mean = sexual_mean[-1]
        final_asexual_mean = asexual_mean[-1]
        final_advantage = sexual_advantage[-1]
        final_advantage_pct = sexual_advantage_pct[-1]
        final_p_value = p_values[-1]
        
        print(f"Final Generation Results:")
        print(f"  Sexual reproduction:   {final_sexual_mean:.3f} ± {sexual_std[-1]:.3f}")
        print(f"  Asexual reproduction:  {final_asexual_mean:.3f} ± {asexual_std[-1]:.3f}")
        print(f"  Sexual advantage:      {final_advantage:+.3f} ({final_advantage_pct:+.1f}%)")
        print(f"  Statistical significance: {'***' if final_p_value < 0.001 else '**' if final_p_value < 0.01 else '*' if final_p_value < 0.05 else 'ns'}")
        
        # Count runs where sexual wins
        sexual_wins = np.sum(sexual_array[:, -1] > asexual_array[:, -1])
        asexual_wins = np.sum(asexual_array[:, -1] > sexual_array[:, -1])
        ties = len(sexual_array) - sexual_wins - asexual_wins
        
        print(f"\nWin/Loss Analysis (Final Generation):")
        print(f"  Sexual wins:  {sexual_wins}/{len(sexual_array)} ({sexual_wins/len(sexual_array)*100:.1f}%)")
        print(f"  Asexual wins: {asexual_wins}/{len(sexual_array)} ({asexual_wins/len(sexual_array)*100:.1f}%)")
        print(f"  Ties:         {ties}/{len(sexual_array)} ({ties/len(sexual_array)*100:.1f}%)")
        
        # Store results for later use
        self.results['sexual'] = sexual_array
        self.results['asexual'] = asexual_array
        self.results['statistics'] = {
            'sexual_mean': sexual_mean,
            'sexual_std': sexual_std,
            'asexual_mean': asexual_mean,
            'asexual_std': asexual_std,
            'sexual_advantage': sexual_advantage,
            'sexual_advantage_pct': sexual_advantage_pct,
            'p_values': p_values
        }
    
    def create_analysis_dataframe(self, sexual_array: np.ndarray, asexual_array: np.ndarray, fitness_name: str) -> pd.DataFrame:
        """Create a comprehensive DataFrame for further analysis."""
        
        # Create long-format DataFrame
        data = []
        
        for run_id in range(len(sexual_array)):
            for i, (point, label) in enumerate(zip(self.decile_points, self.decile_labels)):
                # Sexual data
                data.append({
                    'run_id': run_id,
                    'generation': point,
                    'decile': label,
                    'reproduction_mode': 'sexual',
                    'fitness': sexual_array[run_id, i],
                    'fitness_regime': fitness_name
                })
                
                # Asexual data
                data.append({
                    'run_id': run_id,
                    'generation': point,
                    'decile': label,
                    'reproduction_mode': 'asexual',
                    'fitness': asexual_array[run_id, i],
                    'fitness_regime': fitness_name
                })
        
        df = pd.DataFrame(data)
        
        # Add configuration parameters
        for key, value in self.params.items():
            df[key] = value
        df['n_runs'] = self.n_runs
        df['base_seed'] = self.base_seed
        
        return df

def main():
    """Main function to run the analysis."""
    
    # Create analyzer with default parameters
    analyzer = SexualVsAsexualAnalyzer(
        population_size=100,
        genome_length=20,
        mutation_rate=0.005,
        generations=100,
        n_runs=100,
        base_seed=42
    )
    
    # Run analysis with different fitness regimes
    print("Starting Sexual vs. Asexual Reproduction Analysis")
    print("=" * 80)
    
    # Test with weak selection
    print("\n1. Testing with Weak Selection...")
    df_weak = analyzer.run_comprehensive_analysis(weak_selection, "Weak Selection")
    
    # Test with additive selection
    print("\n2. Testing with Additive Selection...")
    df_additive = analyzer.run_comprehensive_analysis(additive_fitness, "Additive Selection")
    
    # Test with neutral evolution
    print("\n3. Testing with Neutral Evolution...")
    df_neutral = analyzer.run_comprehensive_analysis(neutral_fitness, "Neutral Evolution")
    
    # Combine all results
    df_combined = pd.concat([df_weak, df_additive, df_neutral], ignore_index=True)
    
    # Save results
    output_file = "data/sexual_vs_asexual/sexual_vs_asexual_results.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    print(f"DataFrame shape: {df_combined.shape}")
    
    return df_combined

if __name__ == "__main__":
    df_results = main() 