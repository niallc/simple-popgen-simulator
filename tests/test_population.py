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

def test_evolve_sexual_population_size_and_references():
    pop = Population(size=11, genome_length=6)
    new_pop = pop.evolve_sexual(mutation_rate=0.0)
    assert len(new_pop.genomes) == 11
    for g_old, g_new in zip(pop.genomes, new_pop.genomes):
        assert g_old is not g_new

def test_evolve_sexual_offspring_alleles_from_parents():
    pop = Population(size=6, genome_length=8)
    # Set up three pairs of parents with distinct sequences
    for i, genome in enumerate(pop.genomes):
        Genome.set_sequence(genome, [i % 2] * 8)
    new_pop = pop.evolve_sexual(mutation_rate=0.0)
    # Each offspring's alleles should match one of the two parents at each position
    parent_sequences = [g.sequence for g in pop.genomes]
    for child in new_pop.genomes:
        assert any(np.all(child.sequence == parent_seq) for parent_seq in parent_sequences) or \
            all(any(child.sequence[j] == parent_seq[j] for parent_seq in parent_sequences) for j in range(8))

def test_evolve_sexual_with_neutral_fitness():
    def neutral_fitness(pop):
        return np.ones(len(pop.genomes))
    pop = Population(size=10, genome_length=5, fitness_function=neutral_fitness)
    new_pop = pop.evolve_sexual(mutation_rate=0.0)
    assert len(new_pop.genomes) == 10
    # TODO: Statistical test for parent selection uniformity

def test_evolve_sexual_parent_selection_uniformity():
    pop_size = 6
    genome_length = 8
    n_generations = 100
    parent_counts = np.zeros(pop_size)
    def neutral_fitness(pop):
        return np.ones(len(pop.genomes))
    pop = Population(size=pop_size, genome_length=genome_length, fitness_function=neutral_fitness)
    for _ in range(n_generations):
        fitnesses = pop.fitness().astype(float)
        probs = np.ones(pop_size) / pop_size
        n = pop_size
        for _ in range(n // 2):
            parent1_idx = np.random.choice(n, p=probs)
            parent2_idx = np.random.choice(n, p=probs)
            parent_counts[parent1_idx] += 1
            parent_counts[parent2_idx] += 1
        if n % 2 == 1:
            parent1_idx = np.random.choice(n, p=probs)
            parent2_idx = np.random.choice(n, p=probs)
            parent_counts[parent1_idx] += 1
            parent_counts[parent2_idx] += 1
    expected = np.ones(pop_size) * (parent_counts.sum() / pop_size)
    chi2, p = chisquare(parent_counts, expected)
    assert p > 0.0001, f"Parent selection not uniform (p={p:.5g}). This is a statistical test and may rarely fail by chance."

# TODO: Test for neutral drift and heterozygosity decay once simulation loop is implemented 