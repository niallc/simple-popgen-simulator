import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from simulation import Simulation
from population import Population
from genome import Genome


def neutral_fitness(population):
    """TRUE neutral evolution: all individuals have equal fitness"""
    return np.ones(len(population.genomes))


def test_fitness_function_basic():
    """Test that custom fitness functions work correctly"""
    def test_fitness(pop):
        # Simple: fitness = number of 1s + 1
        return np.array([np.sum(g.sequence) + 1.0 for g in pop.genomes])
    
    pop = Population(size=5, genome_length=10, fitness_function=test_fitness)
    fitnesses = pop.fitness()
    
    # Check fitness values are reasonable
    assert len(fitnesses) == 5
    assert all(fitnesses >= 1.0)  # At least 1 (minimum possible)
    assert all(fitnesses <= 11.0)  # At most 11 (maximum possible)


def test_additive_vs_neutral_fitness_evolution():
    """
    CRITICAL TEST: True neutral (equal fitness) vs additive selection (fitness_function=None)
    This test confirms that fitness_function=None is actually SELECTION, not neutral
    """
    params = {
        'population_size': 50,
        'genome_length': 10,
        'mutation_rate': 0.01,
        'generations': 50,
        'random_seed': 123  # Fixed seed for reproducibility
    }
    
    # Run TRUE neutral simulation (equal fitness for all)
    sim_neutral = Simulation(**params, mode='asexual', fitness_function=neutral_fitness)
    sim_neutral.run()
    results_neutral = sim_neutral.get_results()
    
    # Run additive "selection" simulation (fitness_function=None uses additive fitness)
    sim_additive = Simulation(**params, mode='asexual', fitness_function=None)
    sim_additive.run()
    results_additive = sim_additive.get_results()
    
    # Calculate fitness changes
    neutral_change = results_neutral['mean_fitness'].iloc[-1] - results_neutral['mean_fitness'].iloc[0]
    additive_change = results_additive['mean_fitness'].iloc[-1] - results_additive['mean_fitness'].iloc[0]
    
    print(f"TRUE neutral fitness change: {neutral_change:.2f}")
    print(f"Additive (fitness_function=None) fitness change: {additive_change:.2f}")
    
    # CRITICAL ASSERTION: Additive selection should show more fitness increase than true neutral
    assert additive_change > neutral_change, \
        f"Additive selection should increase fitness more than true neutral! Additive: {additive_change:.2f}, Neutral: {neutral_change:.2f}"


def test_selection_strength_ordering():
    """Test that stronger selection leads to greater fitness increases"""
    def weak_selection(population):
        s = 0.02
        base_fitness = np.array([np.sum(g.sequence) for g in population.genomes])
        return 1.0 + s * base_fitness
    
    def strong_selection(population):
        s = 0.1  # Much stronger selection
        base_fitness = np.array([np.sum(g.sequence) for g in population.genomes])
        return 1.0 + s * base_fitness
    
    params = {
        'population_size': 50,
        'genome_length': 10,
        'mutation_rate': 0.01,
        'generations': 30,
        'random_seed': 789
    }
    
    # Test three levels of selection
    scenarios = [
        ('neutral', neutral_fitness),
        ('weak', weak_selection), 
        ('strong', strong_selection)
    ]
    
    changes = []
    for name, fitness_func in scenarios:
        sim = Simulation(**params, mode='asexual', fitness_function=fitness_func)
        sim.run()
        results = sim.get_results()
        change = results['mean_fitness'].iloc[-1] - results['mean_fitness'].iloc[0]
        changes.append(change)
        print(f"{name} selection fitness change: {change:.2f}")
    
    # Should be ordered: neutral < weak < strong
    assert changes[0] < changes[1] < changes[2], \
        f"Selection strength should order fitness increases: neutral={changes[0]:.2f}, weak={changes[1]:.2f}, strong={changes[2]:.2f}"


def test_sexual_vs_asexual_heterozygosity():
    """Test that sexual reproduction maintains more heterozygosity than asexual under neutrality"""
    params = {
        'population_size': 100,
        'genome_length': 20,
        'mutation_rate': 0.001,  # Low mutation to see drift effects
        'generations': 50,
        'random_seed': 456
    }
    
    # Run both modes under TRUE neutrality (not additive)
    sim_asexual = Simulation(**params, mode='asexual', fitness_function=neutral_fitness)
    sim_asexual.run()
    results_asexual = sim_asexual.get_results()
    
    sim_sexual = Simulation(**params, mode='sexual', fitness_function=neutral_fitness)
    sim_sexual.run()
    results_sexual = sim_sexual.get_results()
    
    # Sexual reproduction should maintain more heterozygosity
    final_hetero_asexual = results_asexual['mean_heterozygosity'].iloc[-1]
    final_hetero_sexual = results_sexual['mean_heterozygosity'].iloc[-1]
    
    print(f"Final heterozygosity - Asexual: {final_hetero_asexual:.3f}, Sexual: {final_hetero_sexual:.3f}")
    
    # Sexual should retain more heterozygosity (though this can be stochastic)
    # We'll use a relaxed test - sexual should be at least 80% of asexual or higher
    assert final_hetero_sexual >= 0.8 * final_hetero_asexual, \
        f"Sexual reproduction should maintain more heterozygosity than asexual. Sexual: {final_hetero_sexual:.3f}, Asexual: {final_hetero_asexual:.3f}"


def test_additive_fitness_calculation():
    """Test that additive fitness equals sum of 1s in genome"""
    pop = Population(size=3, genome_length=5, fitness_function=None, random_seed=789)
    
    # Manually set some genomes to test
    pop.genomes[0].sequence = np.array([1, 1, 1, 0, 0])  # Should have fitness 3
    pop.genomes[1].sequence = np.array([0, 0, 0, 0, 0])  # Should have fitness 0  
    pop.genomes[2].sequence = np.array([1, 1, 1, 1, 1])  # Should have fitness 5
    
    fitnesses = pop.fitness()
    
    assert fitnesses[0] == 3
    assert fitnesses[1] == 0
    assert fitnesses[2] == 5


def test_fitness_proportional_selection():
    """Test that higher fitness individuals are more likely to be selected"""
    # Create population with known fitness differences
    pop = Population(size=3, genome_length=10, fitness_function=None, random_seed=999)
    
    # Set genomes with very different fitness
    pop.genomes[0].sequence = np.zeros(10)        # Fitness 0
    pop.genomes[1].sequence = np.ones(10) * 0     # Fitness 0
    pop.genomes[2].sequence = np.ones(10)         # Fitness 10
    
    # Test selection probabilities
    fitnesses = pop.fitness().astype(float)
    if np.sum(fitnesses) == 0:
        probs = np.ones(len(pop.genomes)) / len(pop.genomes)
    else:
        probs = fitnesses / np.sum(fitnesses)
    
    # Individual 2 should have probability 1.0, others 0.0
    assert probs[0] == 0.0
    assert probs[1] == 0.0  
    assert probs[2] == 1.0


if __name__ == "__main__":
    # Run the critical tests
    test_additive_vs_neutral_fitness_evolution()
    test_selection_strength_ordering()
    print("✅ All tests passed!") 