import numpy as np
from sklearn.utils import shuffle

def initial_build(centroids, drone_limit):
    '''
    Generates a vector of vertices, with a graph of 3+-2.
    A Graph(V,E) will now be called a chromosome.

    Parameters
    ----------
    centroids : [float,float]
        A position vector of centroids.

    Returns
    -------
    [[int]]
        A matrix, of 1xn, where each n element is a chromosome.
    '''
    centroids = shuffle(centroids, random_state=0)
    div = max(round(len(centroids) / drone_limit + 0.0001 - 0.5), 3)  # Ensure div >= 3
    n = round(len(centroids) / div + 0.0001 - 0.5)
    chromosome = [[] for _ in range(n)]
    loop = 0
    for point in centroids:
        chromosome[loop].append(point)
        loop = (loop + 1) % n  # Loop over the chromosomes
    return chromosome


def reverse_section(chromosome):
    '''
    Reverse a random section of a chromosome.

    Parameters
    ----------
    chromosome : list
        A chromosome representing a sequence of points.

    Returns
    -------
    list
        The modified chromosome with a reversed section.
    '''
    n = len(chromosome)
    if n < 3:
        return chromosome  # If chromosome is too short, skip mutation
    ini = int(np.random.uniform(0, n - 3))  # Starting index
    end = int(np.random.uniform(ini, n - 1))  # Ending index
    chromosome[ini:end] = chromosome[ini:end][::-1]  # Reverse the section
    return chromosome


def swap_positions(chromosome):
    '''
    Swaps two random positions in a chromosome.

    Parameters
    ----------
    chromosome : list
        A chromosome representing a sequence of points.

    Returns
    -------
    list
        The modified chromosome with two points swapped.
    '''
    n = len(chromosome)
    if n < 2:
        return chromosome  # If chromosome is too short, skip mutation
    pos = np.random.choice(n, 2, replace=False)  # Select two distinct positions
    chromosome[pos[0]], chromosome[pos[1]] = chromosome[pos[1]], chromosome[pos[0]]  # Swap positions
    return chromosome


def insert_at_different_position(chromosome):
    '''
    Removes a point and inserts it at a different position.

    Parameters
    ----------
    chromosome : list
        A chromosome representing a sequence of points.

    Returns
    -------
    list
        The modified chromosome with a point reinserted.
    '''
    n = len(chromosome)

    # Ensure that the chromosome has at least two points before trying to select positions
    if n < 2:
        return chromosome  # Return the chromosome unchanged if it's too small to mutate

    # Select two distinct positions
    pos = np.random.choice(n, 2, replace=False)
    point = chromosome.pop(pos[0])  # Remove the point
    chromosome.insert(pos[1], point)  # Insert it at a different position

    return chromosome


def mutator_in_chromo(chromosomes):
    '''
    Generates 3 mutated chromosome sets using in-chromosome methods.

    Parameters
    ----------
    chromosomes : list of lists
        A set of chromosomes (each representing a path).

    Returns
    -------
    list of lists
        3 mutated versions of the input chromosomes.
    '''
    new_chromo = []

    for mutator_func in [reverse_section, swap_positions, insert_at_different_position]:
        tmp_chromosome = chromosomes.copy()
        s = np.random.randint(len(tmp_chromosome))  # Select a chromosome to mutate
        tmp_chromosome[s] = mutator_func(tmp_chromosome[s])  # Apply mutation
        new_chromo.append(tmp_chromosome)

    return new_chromo


def swap_vertices_between_chromosomes(chromosomes):
    '''
    Swaps a segment between two chromosomes.

    Parameters
    ----------
    chromosomes : list of lists
        A set of chromosomes (each representing a path).

    Returns
    -------
    list
        A new set of chromosomes with swapped segments.
    '''
    tmp_chromosome = chromosomes.copy()
    pos = np.random.choice(len(chromosomes), 2, replace=False)  # Choose two distinct chromosomes
    chromo1, chromo2 = tmp_chromosome[pos[0]], tmp_chromosome[pos[1]]

    if len(chromo1) < 2 or len(chromo2) < 2:  # Ensure there are enough points to swap
        return tmp_chromosome

    # Choose random segments to swap
    ini1, end1 = sorted(np.random.choice(len(chromo1), 2, replace=False))
    ini2, end2 = sorted(np.random.choice(len(chromo2), 2, replace=False))

    chromo1_segment = chromo1[ini1:end1]
    chromo2_segment = chromo2[ini2:end2]

    # Swap the segments
    tmp_chromosome[pos[0]] = chromo1[:ini1] + chromo2_segment + chromo1[end1:]
    tmp_chromosome[pos[1]] = chromo2[:ini2] + chromo1_segment + chromo2[end2:]

    return tmp_chromosome


def merge_chromosomes(chromosomes):
    '''
    Merges two chromosomes into one.

    Parameters
    ----------
    chromosomes : list of lists
        A set of chromosomes (each representing a path).

    Returns
    -------
    list
        A new set of chromosomes with two merged into one.
    '''
    tmp_chromosome = chromosomes.copy()
    if len(tmp_chromosome) < 2:
        return tmp_chromosome  # No merge possible with fewer than 2 chromosomes

    pos = np.random.choice(len(tmp_chromosome), 2, replace=False)
    chromo1 = tmp_chromosome.pop(pos[0])
    chromo2 = tmp_chromosome.pop(pos[1] - 1 if pos[1] > pos[0] else pos[1])

    # Merge the two chromosomes
    tmp_chromosome.append(chromo1 + chromo2)

    return tmp_chromosome


def split_chromosome(chromosomes):
    '''
    Splits a chromosome into two separate chromosomes.

    Parameters
    ----------
    chromosomes : list of lists
        A set of chromosomes (each representing a path).

    Returns
    -------
    list
        A new set of chromosomes with one chromosome split into two.
    '''
    tmp_chromosome = chromosomes.copy()
    s = np.random.randint(len(chromosomes))  # Select a chromosome to split
    chromo = chromosomes[s]

    if len(chromo) < 6:  # Ensure the chromosome has enough points to split
        return tmp_chromosome

    # Randomly choose a section to split
    ini, end = sorted(np.random.choice(len(chromo), 2, replace=False))
    while end - ini < 3:  # Ensure the split section is large enough
        ini, end = sorted(np.random.choice(len(chromo), 2, replace=False))

    chromo1 = chromo[ini:end]
    chromo2 = chromo[:ini] + chromo[end:]

    # Replace the original chromosome with the two new ones
    tmp_chromosome.pop(s)
    tmp_chromosome.extend([chromo1, chromo2])

    return tmp_chromosome


def mutator_cross_chromo(chromosomes):
    '''
    Generates new children from a set of chromosomes using cross-chromosome methods.

    Parameters
    ----------
    chromosomes : list of lists
        A set of chromosomes (each representing a path).

    Returns
    -------
    list of lists
        A new set of chromosomes after applying cross-chromosome mutations.
    '''
    new_chromo = []

    for cross_func in [swap_vertices_between_chromosomes, merge_chromosomes, split_chromosome]:
        new_chromo.append(cross_func(chromosomes.copy()))

    return new_chromo


if __name__ == "__main__":
    # Example test cases to validate the mutators
    centroids = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    chromosomes = initial_build(centroids, drone_limit=3)

    print("Initial Chromosomes:")
    print(chromosomes)

    # Testing in-chromosome mutations
    mutated_chromosomes = mutator_in_chromo(chromosomes)
    print("\nIn-Chromosome Mutations:")
    for chromo in mutated_chromosomes:
        print(chromo)

    # Testing cross-chromosome mutations
    cross_mutations = mutator_cross_chromo(chromosomes)
    print("\nCross-Chromosome Mutations:")
    for chromo in cross_mutations:
        print(chromo)
