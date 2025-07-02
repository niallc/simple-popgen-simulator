import numpy as np
from genome import Genome

class Population:
    def __init__(self, size, genome_length, fitness_function=None, random_seed=None):
        """
        Initialize a population.
        
        Args:
            size: Number of individuals in the population
            genome_length: Length of each genome
            fitness_function: Function to calculate fitness. If None, uses additive fitness.
                             Must accept a Population instance and return numpy array.
            random_seed: Optional random seed for reproducibility.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.genomes = [Genome(genome_length=genome_length) for _ in range(size)]
        self.fitness_function = fitness_function
        # Enforce fitness function signature if custom
        if fitness_function is not None:
            if not callable(fitness_function):
                raise ValueError("fitness_function must be callable")
            import inspect
            sig = inspect.signature(fitness_function)
            if len(sig.parameters) != 1:
                raise ValueError("fitness_function must take exactly one argument (the population)")
            # Test return type and length
            test_result = fitness_function(self)
            if not isinstance(test_result, np.ndarray):
                raise ValueError("fitness_function must return a numpy array")
            if len(test_result) != size:
                raise ValueError("fitness_function must return an array of length equal to population size")

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

    def evolve_sexual(self, mutation_rate):
        fitnesses = self.fitness().astype(float)
        if np.sum(fitnesses) == 0:
            probs = np.ones(len(self.genomes)) / len(self.genomes)
        else:
            probs = fitnesses / np.sum(fitnesses)
        new_genomes = []
        n = len(self.genomes)
        for _ in range(n // 2):
            parent1_idx = np.random.choice(n, p=probs)
            parent2_idx = np.random.choice(n, p=probs)
            parent1 = self.genomes[parent1_idx]
            parent2 = self.genomes[parent2_idx]
            # Each pair produces 2 offspring
            child1 = parent1.crossover(parent2)
            child2 = parent2.crossover(parent1)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            new_genomes.extend([child1, child2])
        # If n is odd, add one more offspring
        if len(new_genomes) < n:
            parent1_idx = np.random.choice(n, p=probs)
            parent2_idx = np.random.choice(n, p=probs)
            parent1 = self.genomes[parent1_idx]
            parent2 = self.genomes[parent2_idx]
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate)
            new_genomes.append(child)
        new_pop = Population(size=n, genome_length=self.genomes[0].sequence.size, fitness_function=self.fitness_function)
        new_pop.genomes = new_genomes
        return new_pop 