import numpy as np
from src.genome import Genome

def test_genome_creation():
    genome_length = 10
    g = Genome(genome_length=genome_length)
    assert isinstance(g.sequence, np.ndarray)
    assert g.sequence.shape == (genome_length,)
    assert set(np.unique(g.sequence)).issubset({0, 1})

def test_genome_mutation():
    genome_length = 20
    g = Genome(genome_length=genome_length)
    original = g.sequence.copy()
    g.mutate(mutation_rate=1.0)  # All bits should flip
    assert np.all(g.sequence != original)
    g2 = Genome(genome_length=genome_length)
    original2 = g2.sequence.copy()
    g2.mutate(mutation_rate=0.0)  # No bits should flip
    assert np.array_equal(g2.sequence, original2)

def test_genome_crossover():
    genome_length = 10
    g1 = Genome(genome_length=genome_length)
    g2 = Genome(genome_length=genome_length)
    # Set deterministic, different parent sequences
    Genome.set_sequence(g1, [0]*genome_length)
    Genome.set_sequence(g2, [1]*genome_length)
    child = g1.crossover(g2)
    assert isinstance(child, Genome)
    assert child.sequence.shape == (genome_length,)
    # Child should have a prefix of 0s and a suffix of 1s (or vice versa, depending on crossover point)
    # Check that each position matches one of the parents
    for i in range(genome_length):
        assert child.sequence[i] == 0 or child.sequence[i] == 1
        assert child.sequence[i] == g1.sequence[i] or child.sequence[i] == g2.sequence[i] 