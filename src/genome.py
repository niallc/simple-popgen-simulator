import numpy as np

class Genome:
    def __init__(self, genome_length):
        self.sequence = np.random.randint(0, 2, size=genome_length, dtype=np.int8) 

    def mutate(self, mutation_rate):
        mask = np.random.rand(self.sequence.size) < mutation_rate
        self.sequence[mask] = 1 - self.sequence[mask]

    def crossover(self, other):
        assert self.sequence.size == other.sequence.size
        length = self.sequence.size
        if length < 2:
            return Genome(genome_length=length)
        point = np.random.randint(1, length)
        child_seq = np.concatenate([self.sequence[:point], other.sequence[point:]])
        child = Genome(genome_length=length)
        child.sequence = child_seq.copy()
        return child

    @staticmethod
    def set_sequence(genome, sequence):
        genome.sequence = np.array(sequence, dtype=np.int8) 

    def __init__(self, size, genome_length, fitness_function=None):
        """
        Initialize a population.
        
        Args:
            size: Number of individuals in the population
            genome_length: Length of each genome
            fitness_function: Function to calculate fitness. If None, uses additive fitness.
                             Must accept a Population instance and return numpy array.
        
        Note:
            Using fitness_function=None (default) is fine for tests and simple cases.
            For production code, consider explicitly specifying your fitness function
            to make the behavior clear and avoid potential confusion.
        """ 