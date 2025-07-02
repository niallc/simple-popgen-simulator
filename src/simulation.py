import pandas as pd
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

    # Main loop and results collection to be implemented in the next step 