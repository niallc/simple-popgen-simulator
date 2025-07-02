import numpy as np
from src.genome import Genome

class Population:
    def __init__(self, size, genome_length, fitness_function=None):
        """
        Initialize a population.
        
        Args:
            size: Number of individuals in the population
            genome_length: Length of each genome
            fitness_function: Function to calculate fitness. If None, uses additive fitness.
                             Must accept a Population instance and return numpy array.
        """
        self.genomes = [Genome(genome_length=genome_length) for _ in range(size)]
        self.fitness_function = fitness_function

    def additive_fitness(self):
        # Simple additive fitness: sum of 1s in the genome
        return np.array([np.sum(g.sequence) for g in self.genomes])

    def fitness(self):
        if self.fitness_function is None:
            return self.additive_fitness()
        else:
            return self.fitness_function(self)

    def evolve_asexual(self, mutation_rate):
        fitnesses = self.fitness().astype(float)
        # Avoid division by zero: if all fitnesses are zero, use uniform probabilities
        if np.sum(fitnesses) == 0:
            probs = np.ones(len(self.genomes)) / len(self.genomes)
        else:
            probs = fitnesses / np.sum(fitnesses)
        new_genomes = []
        for _ in range(len(self.genomes)):
            parent_idx = np.random.choice(len(self.genomes), p=probs)
            parent = self.genomes[parent_idx]
            child = Genome(genome_length=len(parent.sequence))
            child.sequence = parent.sequence.copy()
            child.mutate(mutation_rate)
            new_genomes.append(child)
        new_pop = Population(size=len(self.genomes), genome_length=len(self.genomes[0].sequence), fitness_function=self.fitness_function)
        new_pop.genomes = new_genomes
        return new_pop

    def tournament_selection(self, tournament_size):
        tournament_indices = np.random.choice(len(self.genomes), tournament_size, replace=False)
        tournament_genomes = [self.genomes[i] for i in tournament_indices]
        return np.random.choice(tournament_genomes) 