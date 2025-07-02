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