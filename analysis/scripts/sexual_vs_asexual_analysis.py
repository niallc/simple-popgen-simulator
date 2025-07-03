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
from datetime import datetime
import os
import re
import sys

# Simple, hardcoded project root - clean and obvious
project_root = os.path.expanduser("~/Documents/programming/SelectionAndSexualReproduction")

# Add project root to Python path for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the core simulation module
from src.simulation import Simulation

# Set up data directories relative to project root
data_dir = os.path.join(project_root, 'analysis', 'data', 'sexual_vs_asexual')
archive_dir = os.path.join(project_root, 'analysis', 'data', 'archive')

# Ensure data directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(archive_dir, exist_ok=True)

# Define fitness functions for the new two-function approach
def neutral_reproduction(population):
    """Neutral evolution: all individuals have equal reproductive fitness"""
    return np.ones(len(population.genomes))

def weak_selection_reproduction(population):
    """Weak selection on reproductive fitness: fitness = 1 + s * (number of 1s)"""
    s = 0.05  # weak selection coefficient
    base_fitness = np.array([np.sum(g.sequence) for g in population.genomes])
    return 1.0 + s * base_fitness

def additive_reproduction(population):
    """Additive selection on reproductive fitness: fitness = sum of 1s in genome"""
    return np.array([np.sum(g.sequence) for g in population.genomes])

def genome_complexity_analysis(population):
    """Analysis trait: genome complexity (sum of 1s) - useful for measuring drift"""
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
        
    def run_single_comparison(self, run_id: int, reproductive_fitness_function, analysis_trait_function=None) -> Tuple[Dict, Dict]:
        """
        Run a single comparison between sexual and asexual reproduction.
        
        Args:
            run_id: Run identifier for seed generation
            reproductive_fitness_function: Function that drives selection
            analysis_trait_function: Function for measuring traits (default: additive fitness)
            
        Returns:
            Tuple of (sexual_results, asexual_results)
        """
        # Use different seeds for each run but keep them systematic for repeatable comparisons
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
            reproductive_fitness_function=reproductive_fitness_function,
            analysis_trait_function=analysis_trait_function
        )
        sexual_sim.run()
        sexual_results = sexual_sim.get_results()
        
        # Run asexual simulation
        asexual_sim = Simulation(
            **asexual_params, 
            mode='asexual', 
            reproductive_fitness_function=reproductive_fitness_function,
            analysis_trait_function=analysis_trait_function
        )
        asexual_sim.run()
        asexual_results = asexual_sim.get_results()
        
        return sexual_results, asexual_results
    
    def normalize_fitness(self, fitness_series: pd.Series) -> pd.Series:
        """Normalize fitness to start at 1.0 for fair comparison."""
        initial_fitness = fitness_series.iloc[0]
        return fitness_series / initial_fitness
    
    def extract_decile_fitness(self, results: Dict) -> List[float]:
        """Extract analysis trait values at decile points and normalize."""
        # Use mean_analysis_trait instead of mean_fitness
        trait_series = results['mean_analysis_trait']
        normalized_trait = self.normalize_fitness(trait_series)
        
        decile_fitness = []
        for point in self.decile_points:
            if point < len(normalized_trait):
                decile_fitness.append(normalized_trait.iloc[point])
            else:
                decile_fitness.append(normalized_trait.iloc[-1])
        
        return decile_fitness
    
    def run_comprehensive_analysis(self, reproductive_fitness_function, analysis_trait_function=None, fitness_name: str = "Custom"):
        """
        Run comprehensive analysis comparing sexual vs. asexual reproduction.
        
        Args:
            reproductive_fitness_function: Function that drives selection
            analysis_trait_function: Function for measuring traits (default: additive fitness)
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
                sexual_results, asexual_results = self.run_single_comparison(run_id, reproductive_fitness_function, analysis_trait_function)
                
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

def generate_filename(analyzer, fitness_name: str) -> str:
    """
    Generate a descriptive filename with parameters and timestamp.
    
    Args:
        analyzer: SexualVsAsexualAnalyzer instance
        fitness_name: Name of the fitness regime
        
    Returns:
        str: Descriptive filename
    """
    # Extract key parameters
    pop_size = analyzer.params['population_size']
    genome_len = analyzer.params['genome_length']
    mut_rate = analyzer.params['mutation_rate']
    generations = analyzer.params['generations']
    n_runs = analyzer.n_runs
    
    # Create parameter string
    param_str = f"pop{pop_size}_gen{genome_len}_mut{mut_rate:.3f}_gens{generations}_runs{n_runs}"
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Create fitness regime string (sanitized for filename)
    fitness_str = fitness_name.lower().replace(" ", "_").replace("-", "_")
    
    # Combine into filename
    filename = f"sexual_vs_asexual_{fitness_str}_{param_str}_{timestamp}.csv"
    
    return filename

def get_versioned_filename(base_path: str) -> str:
    """
    Generate a versioned filename to prevent clobbering existing files.
    
    Args:
        base_path: The intended file path (e.g., "data/file.csv")
        
    Returns:
        str: A unique filename that won't clobber existing files
        
    Examples:
        - If "data/file.csv" doesn't exist → returns "data/file.csv"
        - If "data/file.csv" exists → returns "data/file_v2.csv"
        - If "data/file.csv" and "data/file_v2.csv" exist → returns "data/file_v3.csv"
    """
    # Check if the base file exists
    if not os.path.exists(base_path):
        return base_path
    
    # Extract directory, filename, and extension
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    # Pattern to match versioned files: name_vN.ext
    version_pattern = re.compile(rf'^{re.escape(name)}_v(\d+){re.escape(ext)}$')
    
    # Find all existing versioned files
    existing_versions = []
    if os.path.exists(directory):
        try:
            for file in os.listdir(directory):
                match = version_pattern.match(file)
                if match:
                    version_num = int(match.group(1))
                    existing_versions.append(version_num)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not read directory {directory}: {e}")
            # Fall back to adding timestamp to avoid clobbering
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_path}.{timestamp}"
    
    # Find the next version number
    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        next_version = 2  # Start with v2 since v1 is the base file
    
    # Construct the new filename
    versioned_filename = f"{name}_v{next_version}{ext}"
    versioned_path = os.path.join(directory, versioned_filename)
    
    return versioned_path

def list_file_versions(base_path: str) -> None:
    """
    List all existing versions of a file and show what the next version would be.
    
    Args:
        base_path: The base file path to check for versions
    """
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    print(f"📁 File version analysis for: {base_path}")
    
    # Check if base file exists
    if os.path.exists(base_path):
        print(f"✅ Base file exists: {filename}")
    else:
        print(f"❌ Base file does not exist: {filename}")
    
    # Pattern to match versioned files
    version_pattern = re.compile(rf'^{re.escape(name)}_v(\d+){re.escape(ext)}$')
    
    # Find all existing versioned files
    existing_versions = []
    if os.path.exists(directory):
        try:
            for file in os.listdir(directory):
                match = version_pattern.match(file)
                if match:
                    version_num = int(match.group(1))
                    existing_versions.append(version_num)
        except (OSError, PermissionError) as e:
            print(f"❌ Could not read directory {directory}: {e}")
            return
    
    if existing_versions:
        existing_versions.sort()
        print(f"📋 Existing versions: {existing_versions}")
        print(f"📋 Versioned files:")
        for v in existing_versions:
            versioned_file = f"{name}_v{v}{ext}"
            versioned_path = os.path.join(directory, versioned_file)
            if os.path.exists(versioned_path):
                size = os.path.getsize(versioned_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(versioned_path)).strftime("%Y-%m-%d %H:%M")
                print(f"   v{v}: {versioned_file} ({size:,} bytes, {mtime})")
    else:
        print("📋 No versioned files found")
    
    # Show what the next version would be
    next_version = get_versioned_filename(base_path)
    if next_version != base_path:
        print(f"🔄 Next version would be: {os.path.basename(next_version)}")
    else:
        print(f"🔄 No versioning needed (file doesn't exist)")

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
    
    # Test with neutral evolution
    print("\n1. Testing with Neutral Evolution...")
    df_neutral = analyzer.run_comprehensive_analysis(neutral_reproduction, genome_complexity_analysis, "Neutral Evolution")

    # Test with weak selection
    print("\n2. Testing with Weak Selection...")
    df_weak = analyzer.run_comprehensive_analysis(weak_selection_reproduction, genome_complexity_analysis, "Weak Selection")
    
    # Test with additive selection
    print("\n3. Testing with Additive Selection...")
    df_additive = analyzer.run_comprehensive_analysis(additive_reproduction, genome_complexity_analysis, "Additive Selection")
        
    # Combine all results
    df_combined = pd.concat([df_weak, df_additive, df_neutral], ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate descriptive filename
    filename = generate_filename(analyzer, "comprehensive")
    base_output_file = os.path.join(data_dir, filename)
    
    # Get versioned filename to prevent clobbering
    output_file = get_versioned_filename(base_output_file)
    
    # Show versioning information if needed
    if output_file != base_output_file:
        print(f"\n📁 File versioning:")
        print(f"   Base filename: {filename}")
        print(f"   Versioned filename: {os.path.basename(output_file)}")
        print(f"   This prevents clobbering existing files")
    
    # Save results
    df_combined.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    print(f"DataFrame shape: {df_combined.shape}")
    
    return df_combined

if __name__ == "__main__":
    df_results = main() 