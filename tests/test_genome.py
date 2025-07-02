import numpy as np
from src.genome import Genome

def test_genome_creation():
    length = 10
    g = Genome(length=length)
    assert isinstance(g.sequence, np.ndarray)
    assert g.sequence.shape == (length,)
    assert set(np.unique(g.sequence)).issubset({0, 1})

def test_genome_mutation():
    length = 20
    g = Genome(length=length)
    original = g.sequence.copy()
    g.mutate(mutation_rate=1.0)  # All bits should flip
    assert np.all(g.sequence != original)
    g2 = Genome(length=length)
    original2 = g2.sequence.copy()
    g2.mutate(mutation_rate=0.0)  # No bits should flip
    assert np.array_equal(g2.sequence, original2)

def test_genome_crossover():
    length = 10
    g1 = Genome(length=length)
    g2 = Genome(length=length)
    # Set deterministic, different parent sequences
    Genome.set_sequence(g1, [0]*length)
    Genome.set_sequence(g2, [1]*length)
    child = g1.crossover(g2)
    assert isinstance(child, Genome)
    assert child.sequence.shape == (length,)
    # Child should have a prefix of 0s and a suffix of 1s (or vice versa, depending on crossover point)
    # Check that each position matches one of the parents
    for i in range(length):
        assert child.sequence[i] == 0 or child.sequence[i] == 1
        assert child.sequence[i] == g1.sequence[i] or child.sequence[i] == g2.sequence[i] 