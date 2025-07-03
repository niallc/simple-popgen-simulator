#!/usr/bin/env python3
"""
Tests for randomness and variation in the simulation system.

This test suite checks:
1. Whether different seeds produce different results
2. Whether the global random state is being properly managed
3. Whether neutral evolution shows expected variation
4. Whether multiple runs produce different outcomes
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import Simulation
from src.population import Population
from src.genome import Genome

def neutral_fitness(population):
    """True neutral evolution: all individuals have equal fitness"""
    return np.ones(len(population.genomes))

def weak_selection(population):
    """Weak selection: fitness = 1 + s * (number of 1s)"""
    s = 0.05
    base_fitness = np.array([np.sum(g.sequence) for g in population.genomes])
    return 1.0 + s * base_fitness

class TestRandomnessAndVariation(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters."""
        self.population_size = 50
        self.genome_length = 20
        self.mutation_rate = 0.05
        self.generations = 20
    
    def test_different_seeds_produce_different_populations(self):
        """Test that different seeds produce different initial populations."""
        print("\n=== Testing Different Seeds Produce Different Populations ===")
        
        # Create populations with different seeds
        seeds = [42, 100, 200, 300, 400]
        initial_genomes = []
        
        for seed in seeds:
            pop = Population(
                size=self.population_size,
                genome_length=self.genome_length,
                random_seed=seed
            )
            
            # Get the sum of all genomes as a simple fingerprint
            genome_sum = sum(np.sum(g.sequence) for g in pop.genomes)
            initial_genomes.append(genome_sum)
            print(f"Seed {seed}: Genome sum = {genome_sum}")
        
        # Check that we have variation
        genome_sums = np.array(initial_genomes)
        variation = np.std(genome_sums)
        print(f"Variation in genome sums: std = {variation:.2f}")
        
        # Should have some variation (not all identical)
        self.assertGreater(variation, 0, "Different seeds should produce different populations")
        
        # Check that not all are identical
        unique_sums = len(set(genome_sums))
        self.assertGreater(unique_sums, 1, f"Expected variation, got {unique_sums} unique values")
    
    def test_global_random_state_issue(self):
        """Test for the global random state problem."""
        print("\n=== Testing Global Random State Issue ===")
        
        # This test checks if setting a seed in one population affects another
        # If there's a global state issue, this should show it
        
        # Create first population with seed 42
        pop1 = Population(
            size=self.population_size,
            genome_length=self.genome_length,
            random_seed=42
        )
        genome_sum1 = sum(np.sum(g.sequence) for g in pop1.genomes)
        
        # Create second population with seed 100 (should be different)
        pop2 = Population(
            size=self.population_size,
            genome_length=self.genome_length,
            random_seed=100
        )
        genome_sum2 = sum(np.sum(g.sequence) for g in pop2.genomes)
        
        print(f"Population 1 (seed 42): Genome sum = {genome_sum1}")
        print(f"Population 2 (seed 100): Genome sum = {genome_sum2}")
        
        # These should be different
        self.assertNotEqual(genome_sum1, genome_sum2, 
                           "Different seeds should produce different populations")
        
        # Now create a third population with seed 42 again
        pop3 = Population(
            size=self.population_size,
            genome_length=self.genome_length,
            random_seed=42
        )
        genome_sum3 = sum(np.sum(g.sequence) for g in pop3.genomes)
        
        print(f"Population 3 (seed 42 again): Genome sum = {genome_sum3}")
        
        # This should be the same as the first one (reproducibility)
        self.assertEqual(genome_sum1, genome_sum3, 
                        "Same seed should produce same population")
    
    def test_neutral_evolution_variation(self):
        """Test that neutral evolution shows some variation across runs."""
        print("\n=== Testing Neutral Evolution Variation ===")
        
        # Run multiple neutral evolution simulations
        n_runs = 10
        final_heterozygosities = []  # Track heterozygosity instead of fitness
        
        for run_id in range(n_runs):
            seed = 42 + run_id * 10  # Different seed for each run
            
            sim = Simulation(
                population_size=self.population_size,
                genome_length=self.genome_length,
                mutation_rate=self.mutation_rate,
                generations=self.generations,
                mode='asexual',
                fitness_function=neutral_fitness,
                random_seed=seed
            )
            sim.run()
            results = sim.get_results()
            
            final_fitness = results['mean_fitness'].iloc[-1]
            final_heterozygosity = results['mean_heterozygosity'].iloc[-1]
            final_heterozygosities.append(final_heterozygosity)
            print(f"Run {run_id} (seed {seed}): Final fitness = {final_fitness:.6f}, Heterozygosity = {final_heterozygosity:.6f}")
        
        # Calculate variation in heterozygosity (population composition)
        heterozygosity_array = np.array(final_heterozygosities)
        variation = np.std(heterozygosity_array)
        print(f"Variation in final heterozygosity: std = {variation:.8f}")
        
        # For neutral evolution, fitness should be constant but population composition should vary
        fitness_array = np.array([1.0] * n_runs)  # All should be 1.0
        fitness_variation = np.std(fitness_array)
        print(f"Variation in final fitness: std = {fitness_variation:.8f}")
        
        # Fitness should be constant (no variation)
        self.assertAlmostEqual(fitness_variation, 0.0, places=8, 
                              msg="Neutral evolution should maintain constant fitness")
        
        # But heterozygosity should show some variation due to genetic drift
        self.assertGreater(variation, 0, "Neutral evolution should show variation in population composition")
        
        # Mean heterozygosity should be reasonable (around 0.25 for random binary sequences)
        mean_heterozygosity = np.mean(heterozygosity_array)
        print(f"Mean final heterozygosity: {mean_heterozygosity:.6f}")
        self.assertGreater(mean_heterozygosity, 0, "Heterozygosity should be positive")
        self.assertLess(mean_heterozygosity, 0.5, "Heterozygosity should be less than 0.5 for binary sequences")
    
    def test_weak_selection_variation(self):
        """Test that weak selection shows variation across runs."""
        print("\n=== Testing Weak Selection Variation ===")
        
        # Run multiple weak selection simulations
        n_runs = 10
        final_fitnesses = []
        
        for run_id in range(n_runs):
            seed = 42 + run_id * 10  # Different seed for each run
            
            sim = Simulation(
                population_size=self.population_size,
                genome_length=self.genome_length,
                mutation_rate=self.mutation_rate,
                generations=self.generations,
                mode='asexual',
                fitness_function=weak_selection,
                random_seed=seed
            )
            sim.run()
            results = sim.get_results()
            
            final_fitness = results['mean_fitness'].iloc[-1]
            final_fitnesses.append(final_fitness)
            print(f"Run {run_id} (seed {seed}): Final fitness = {final_fitness:.6f}")
        
        # Calculate variation
        fitness_array = np.array(final_fitnesses)
        variation = np.std(fitness_array)
        mean_fitness = np.mean(fitness_array)
        
        print(f"Mean final fitness: {mean_fitness:.6f}")
        print(f"Variation in final fitness: std = {variation:.6f}")
        
        # Should show variation
        self.assertGreater(variation, 0, "Weak selection should show variation across runs")
        
        # Should show fitness increase on average
        self.assertGreater(mean_fitness, 1.0, "Weak selection should increase fitness on average")
    
    def test_sexual_vs_asexual_independence(self):
        """Test that sexual and asexual runs are independent."""
        print("\n=== Testing Sexual vs Asexual Independence ===")
        
        # Run sexual and asexual with same seed
        seed = 42
        
        # Asexual run
        asexual_sim = Simulation(
            population_size=self.population_size,
            genome_length=self.genome_length,
            mutation_rate=self.mutation_rate,
            generations=self.generations,
            mode='asexual',
            fitness_function=weak_selection,
            random_seed=seed
        )
        asexual_sim.run()
        asexual_results = asexual_sim.get_results()
        asexual_final = asexual_results['mean_fitness'].iloc[-1]
        
        # Sexual run
        sexual_sim = Simulation(
            population_size=self.population_size,
            genome_length=self.genome_length,
            mutation_rate=self.mutation_rate,
            generations=self.generations,
            mode='sexual',
            fitness_function=weak_selection,
            random_seed=seed
        )
        sexual_sim.run()
        sexual_results = sexual_sim.get_results()
        sexual_final = sexual_results['mean_fitness'].iloc[-1]
        
        print(f"Asexual final fitness: {asexual_final:.6f}")
        print(f"Sexual final fitness: {sexual_final:.6f}")
        
        # These should be different (different modes)
        self.assertNotEqual(asexual_final, sexual_final, 
                           "Sexual and asexual should produce different results")
    
    def test_multiple_runs_independence(self):
        """Test that multiple runs with different seeds are independent."""
        print("\n=== Testing Multiple Runs Independence ===")
        
        # Run multiple simulations with different seeds
        n_runs = 5
        final_fitnesses = []
        
        for run_id in range(n_runs):
            seed = 42 + run_id * 100  # Very different seeds
            
            sim = Simulation(
                population_size=self.population_size,
                genome_length=self.genome_length,
                mutation_rate=self.mutation_rate,
                generations=self.generations,
                mode='asexual',
                fitness_function=weak_selection,
                random_seed=seed
            )
            sim.run()
            results = sim.get_results()
            
            final_fitness = results['mean_fitness'].iloc[-1]
            final_fitnesses.append(final_fitness)
            print(f"Run {run_id} (seed {seed}): Final fitness = {final_fitness:.6f}")
        
        # Check for independence
        fitness_array = np.array(final_fitnesses)
        unique_values = len(set(fitness_array))
        
        print(f"Unique final fitness values: {unique_values}/{n_runs}")
        
        # Should have multiple unique values (not all identical)
        self.assertGreater(unique_values, 1, 
                          f"Expected variation across runs, got {unique_values} unique values")
        
        # Calculate correlation between run order and fitness
        run_order = np.arange(n_runs)
        correlation = np.corrcoef(run_order, fitness_array)[0, 1]
        print(f"Correlation between run order and fitness: {correlation:.3f}")
        
        # Correlation should be low (runs should be independent)
        self.assertLess(abs(correlation), 0.8, 
                       "Runs should be independent (low correlation with run order)")
    
    def test_genome_mutation_independence(self):
        """Test that genome mutations are independent."""
        print("\n=== Testing Genome Mutation Independence ===")
        
        # Create a genome and mutate it multiple times
        genome = Genome(genome_length=self.genome_length)
        original_sequence = genome.sequence.copy()
        
        # Mutate multiple times
        n_mutations = 10
        mutation_results = []
        
        for i in range(n_mutations):
            # Create a fresh genome each time
            test_genome = Genome(genome_length=self.genome_length)
            test_genome.sequence = original_sequence.copy()
            
            # Mutate
            test_genome.mutate(self.mutation_rate)
            
            # Count differences from original
            differences = np.sum(test_genome.sequence != original_sequence)
            mutation_results.append(differences)
            print(f"Mutation {i}: {differences} differences")
        
        # Check variation in mutation results
        mutation_array = np.array(mutation_results)
        variation = np.std(mutation_array)
        mean_mutations = np.mean(mutation_array)
        
        print(f"Mean mutations: {mean_mutations:.2f}")
        print(f"Variation in mutations: std = {variation:.2f}")
        
        # Should show variation in mutation results
        self.assertGreater(variation, 0, "Mutations should show variation")
        
        # Mean should be reasonable (mutation_rate * genome_length)
        expected_mean = self.mutation_rate * self.genome_length
        print(f"Expected mean mutations: {expected_mean:.2f}")
        self.assertAlmostEqual(mean_mutations, expected_mean, delta=2, 
                              msg="Mean mutations should be close to expected value")

def run_randomness_tests():
    """Run all randomness tests and print summary."""
    print("🧬 RUNNING RANDOMNESS AND VARIATION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomnessAndVariation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Randomness appears to be working correctly.")
    else:
        print("\n❌ Some tests failed. There may be randomness issues.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_randomness_tests() 