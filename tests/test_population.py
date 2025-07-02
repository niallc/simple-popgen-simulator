import numpy as np
from src.population import Population
from src.genome import Genome
import pytest
from scipy.stats import chisquare

def test_population_creation():
    pop_size = 5
    genome_length = 8
    pop = Population(size=pop_size, genome_length=genome_length)
    assert len(pop.genomes) == pop_size
    for genome in pop.genomes:
        assert genome.sequence.shape == (genome_length,)
        assert set(np.unique(genome.sequence)).issubset({0, 1})

def test_additive_fitness():
    pop = Population(size=3, genome_length=4)
    # Manually set genomes for predictable fitness
    pop.genomes[0].sequence = np.array([1, 1, 1, 1])  # fitness 4
    pop.genomes[1].sequence = np.array([0, 0, 0, 0])  # fitness 0
    pop.genomes[2].sequence = np.array([1, 0, 1, 0])  # fitness 2
    fitnesses = pop.additive_fitness()
    assert np.array_equal(fitnesses, np.array([4, 0, 2]))

def test_custom_fitness_function():
    def neutral_fitness(population):
        return np.ones(len(population.genomes))
    
    pop = Population(size=3, genome_length=4, fitness_function=neutral_fitness)
    pop.genomes[0].sequence = np.array([1, 1, 1, 1])
    pop.genomes[1].sequence = np.array([0, 0, 0, 0])
    pop.genomes[2].sequence = np.array([1, 0, 1, 0])
    fitnesses = pop.fitness()
    assert np.array_equal(fitnesses, np.array([1, 1, 1]))

def test_wright_fisher_asexual_evolution():
    pop = Population(size=10, genome_length=5)
    # Evolve for one generation
    new_pop = pop.evolve_asexual(mutation_rate=0.0)
    assert len(new_pop.genomes) == 10
    # With mutation_rate=0, all offspring should be exact copies of parents
    parent_sequences = [tuple(g.sequence) for g in pop.genomes]
    offspring_sequences = [tuple(g.sequence) for g in new_pop.genomes]
    for seq in offspring_sequences:
        assert seq in parent_sequences

def test_no_shared_references_after_evolution():
    pop = Population(size=5, genome_length=4)
    new_pop = pop.evolve_asexual(mutation_rate=0.0)
    for g_old, g_new in zip(pop.genomes, new_pop.genomes):
        assert g_old is not g_new

def test_crossover_point_uniformity():
    genome_length = 10
    g1 = Genome(genome_length=genome_length)
    g2 = Genome(genome_length=genome_length)
    Genome.set_sequence(g1, [0]*genome_length)
    Genome.set_sequence(g2, [1]*genome_length)
    n_samples = 1000
    counts = np.zeros(genome_length-1)
    for _ in range(n_samples):
        child = g1.crossover(g2)
        switch_indices = np.where(child.sequence != g1.sequence)[0]
        if len(switch_indices) > 0:
            counts[switch_indices[0]-1] += 1
    expected = np.ones(genome_length-1) * (n_samples / (genome_length-1))
    chi2, p = chisquare(counts, expected)
    assert p > 0.0001, f"Crossover points not uniform (p={p:.5g}). This is a statistical test and may rarely fail by chance."

def test_fitness_function_signature_enforcement():
    # Not callable
    with pytest.raises(ValueError):
        Population(size=3, genome_length=4, fitness_function=42)
    # Wrong number of arguments
    def bad_fitness1():
        return np.ones(3)
    with pytest.raises(ValueError):
        Population(size=3, genome_length=4, fitness_function=bad_fitness1)
    # Wrong return type
    def bad_fitness2(pop):
        return [1, 1, 1]
    with pytest.raises(ValueError):
        Population(size=3, genome_length=4, fitness_function=bad_fitness2)
    # Wrong return length
    def bad_fitness3(pop):
        return np.ones(2)
    with pytest.raises(ValueError):
        Population(size=3, genome_length=4, fitness_function=bad_fitness3)

# TODO: Test for neutral drift and heterozygosity decay once simulation loop is implemented 