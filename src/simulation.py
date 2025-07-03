import pandas as pd
import numpy as np
from src.population import Population

class Simulation:
    def __init__(self, *,
                 population_size,
                 genome_length,
                 mutation_rate,
                 reproductive_fitness_function=None,
                 analysis_trait_function=None,
                 mode=None,  # 'asexual' or 'sexual'
                 generations=100,
                 random_seed=None):
        """
        Initialize the simulation configuration.
        Args:
            population_size: Number of individuals in the population
            genome_length: Length of each genome
            mutation_rate: Per-bit mutation rate
            reproductive_fitness_function: Function that drives selection (default: additive fitness)
            analysis_trait_function: Function for measuring traits (default: additive fitness)
            mode: 'asexual' or 'sexual' (must be specified)
            generations: Number of generations to simulate
            random_seed: Optional random seed for reproducibility
        """
        if mode not in ('asexual', 'sexual'):
            raise ValueError("mode must be 'asexual' or 'sexual'")
        self.population_size = population_size
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.reproductive_fitness_function = reproductive_fitness_function
        self.analysis_trait_function = analysis_trait_function
        self.mode = mode
        self.generations = generations
        self.random_seed = random_seed
        self.population = Population(size=population_size,
                                     genome_length=genome_length,
                                     fitness_function=reproductive_fitness_function,
                                     random_seed=random_seed)
        self.results = []  # Will hold per-generation statistics as dicts

    def run(self):
        """
        Run the simulation for the specified number of generations.
        Collects per-generation statistics: generation, mean reproductive fitness, 
        mean analysis trait, mean heterozygosity.
        """
        self.results = []
        for gen in range(self.generations + 1):
            # Collect reproductive fitness (what drives selection)
            reproductive_fitnesses = self.population.fitness()
            mean_reproductive_fitness = np.mean(reproductive_fitnesses)
            
            # Collect analysis trait (what we measure)
            if self.analysis_trait_function:
                analysis_traits = self.analysis_trait_function(self.population)
            else:
                # Default to additive fitness for analysis trait
                analysis_traits = self.population.additive_fitness()
            mean_analysis_trait = np.mean(analysis_traits)
            
            # Heterozygosity: mean fraction of heterozygous sites across the population
            # For binary genomes, heterozygosity at each site is 2p(1-p), where p is freq of 1s
            genomes_matrix = np.array([g.sequence for g in self.population.genomes])
            p_ones = np.mean(genomes_matrix, axis=0)
            heterozygosity = np.mean(2 * p_ones * (1 - p_ones))
            
            self.results.append({
                'generation': gen,
                'mean_reproductive_fitness': mean_reproductive_fitness,
                'mean_analysis_trait': mean_analysis_trait,
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