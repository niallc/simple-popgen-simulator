import numpy as np
from src.population import Population

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

def test_tournament_selection():
    pop_size = 10
    genome_length = 5
    pop = Population(size=pop_size, genome_length=genome_length)
    
    # Test tournament selection returns a genome
    selected = pop.tournament_selection(tournament_size=3)
    assert selected in pop.genomes
    
    # Test with tournament size equal to population size
    selected2 = pop.tournament_selection(tournament_size=pop_size)
    assert selected2 in pop.genomes
    
    # Test with tournament size 1 (should return random genome)
    selected3 = pop.tournament_selection(tournament_size=1)
    assert selected3 in pop.genomes 