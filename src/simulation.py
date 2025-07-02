import pandas as pd
import numpy as np
from src.population import Population

class Simulation:
    def __init__(self, *,
                 population_size,
                 genome_length,
                 mutation_rate,
                 fitness_function=None,
                 mode=None,  # 'asexual' or 'sexual'
                 generations=100,
                 random_seed=None):
        """
        Initialize the simulation configuration.
        Args:
            population_size: Number of individuals in the population
            genome_length: Length of each genome
            mutation_rate: Per-bit mutation rate
            fitness_function: Function to calculate fitness (default: additive)
            mode: 'asexual' or 'sexual' (must be specified)
            generations: Number of generations to simulate
            random_seed: Optional random seed for reproducibility
        """
        if mode not in ('asexual', 'sexual'):
            raise ValueError("mode must be 'asexual' or 'sexual'")
        self.population_size = population_size
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.mode = mode
        self.generations = generations
        self.random_seed = random_seed
        self.population = Population(size=population_size,
                                     genome_length=genome_length,
                                     fitness_function=fitness_function,
                                     random_seed=random_seed)
        self.results = []  # Will hold per-generation statistics as dicts

    def run(self):
        """
        Run the simulation for the specified number of generations.
        Collects per-generation statistics: generation, mean fitness, mean heterozygosity.
        """
        self.results = []
        for gen in range(self.generations + 1):
            # Collect statistics
            fitnesses = self.population.fitness()
            mean_fitness = np.mean(fitnesses)
            # Heterozygosity: mean fraction of heterozygous sites across the population
            # For binary genomes, heterozygosity at each site is 2p(1-p), where p is freq of 1s
            genomes_matrix = np.array([g.sequence for g in self.population.genomes])
            p_ones = np.mean(genomes_matrix, axis=0)
            heterozygosity = np.mean(2 * p_ones * (1 - p_ones))
            self.results.append({
                'generation': gen,
                'mean_fitness': mean_fitness,
                'mean_heterozygosity': heterozygosity
            })
            # Evolve population (skip on last generation)
            if gen < self.generations:
                if self.mode == 'asexual':
                    self.population = self.population.evolve_asexual(self.mutation_rate)
                elif self.mode == 'sexual':
                    self.population = self.population.evolve_sexual(self.mutation_rate)

    def get_results(self):
        """
        Return the results as a pandas DataFrame.
        """
        return pd.DataFrame(self.results)

    # Main loop and results collection to be implemented in the next step 