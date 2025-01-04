# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:58:48 2024

@author: major
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------
# Simulation Parameters
# ------------------------------

# GA Parameters
NUM_ITERATIONS = 17000      # Number of iterations for GA
POPULATION_SIZE = 100       # Population size for GA
MUTATION_RATE = 0.01        # Mutation rate for GA
CROSSOVER_RATE = 0.8        # Crossover rate for GA
ELITE_SIZE = int(0.05 * POPULATION_SIZE)  # Top 5% as elites

# System Parameters
NUM_UAVS = 10               # Number of UAV-BSs
NUM_USERS_LIST = [80, 200, 450]  # Different numbers of UEs to test
DATA_RATE_CAPACITY = 1e8    # Data rate capacity of each UAV-BS (100 Mbps)

# Data Rate Requirements for UEs (in bps)
DATA_RATE_CHOICES = [5e6, 2e6, 1e6]

# Environment Parameters: (a, b, ηLoS, ηNLoS)
ENVIRONMENTS = {
    "Suburban": (4.88, 0.43, 0.1, 21),
    "Urban": (9.61, 0.43, 0.1, 20),
    "Dense Urban": (12.08, 0.11, 1.6, 23),
    "High-Rise Urban": (27.23, 0.08, 2.3, 34),
}

# ------------------------------
# Utility Functions
# ------------------------------

def initialize_users(num_users):
    """
    Initializes user positions and assigns data rate requirements.

    Parameters:
    - num_users (int): Number of user equipments (UEs).

    Returns:
    - positions (np.ndarray): Array of user positions with shape (num_users, 2).
    - data_rates (np.ndarray): Array of user data rate requirements with shape (num_users,).
    """
    positions = np.random.rand(num_users, 2) * 5000  # UEs in 5000m x 5000m area
    data_rates = np.random.choice(DATA_RATE_CHOICES, num_users)
    return positions, data_rates

def update_user_positions(user_positions, mobility_model='random_walk', speed=10, group_refs=None, user_displacement=None):
    """
    Updates user positions based on the selected mobility model.

    Parameters:
    - user_positions (np.ndarray): Current positions of users with shape (num_users, 2).
    - mobility_model (str): Mobility model to use ('random_walk' or 'rpgm').
    - speed (float): Maximum movement speed per iteration.
    - group_refs (np.ndarray or None): Reference points for groups in RPGM.
    - user_displacement (np.ndarray or None): Displacement vectors for users in RPGM.

    Returns:
    - user_positions (np.ndarray): Updated user positions.
    - group_refs (np.ndarray or None): Updated group reference points (if RPGM).
    """
    if mobility_model == 'random_walk':
        # Random Walk mobility
        angles = np.random.uniform(0, 2 * np.pi, len(user_positions))
        distances = np.random.uniform(0, speed, len(user_positions))
        delta_x = distances * np.cos(angles)
        delta_y = distances * np.sin(angles)
        user_positions[:, 0] += delta_x
        user_positions[:, 1] += delta_y
        user_positions[:, 0] = np.clip(user_positions[:, 0], 0, 5000)
        user_positions[:, 1] = np.clip(user_positions[:, 1], 0, 5000)

    elif mobility_model == 'rpgm':
        # RPGM Mobility
        num_groups = len(group_refs)
        for i in range(num_groups):
            angles = np.random.uniform(0, 2 * np.pi)
            distances = np.random.uniform(0, speed)
            group_refs[i, 0] += distances * np.cos(angles)
            group_refs[i, 1] += distances * np.sin(angles)
            group_refs[i, 0] = np.clip(group_refs[i, 0], 0, 5000)
            group_refs[i, 1] = np.clip(group_refs[i, 1], 0, 5000)

        for user_index in range(len(user_positions)):
            group_index = user_index % num_groups
            user_positions[user_index] = group_refs[group_index] + user_displacement[user_index]
            user_positions[user_index, 0] = np.clip(user_positions[user_index, 0], 0, 5000)
            user_positions[user_index, 1] = np.clip(user_positions[user_index, 1], 0, 5000)

    return user_positions, group_refs

def calculate_path_loss(distance, altitude, env_params):
    """
    Calculates the path loss based on distance, altitude, and environment parameters.

    Parameters:
    - distance (float): Horizontal distance between UAV-BS and UE (in meters).
    - altitude (float): Altitude of the UAV-BS (in meters).
    - env_params (tuple): Environmental parameters (a, b, ηLoS, ηNLoS).

    Returns:
    - path_loss (float): Calculated path loss (in dB).
    """
    a, b, eta_LoS, eta_NLoS = env_params
    # Compute 3D distance
    distance_3d = np.sqrt(distance**2 + altitude**2)
    theta = (180 / np.pi) * np.arctan(altitude / distance)  # Elevation angle in degrees
    p_los = 1 / (1 + a * np.exp(-b * (theta - a)))
    p_nlos = 1 - p_los
    pl_los = 20 * np.log10(distance_3d) + eta_LoS
    pl_nlos = 20 * np.log10(distance_3d) + eta_NLoS
    return pl_los * p_los + pl_nlos * p_nlos

def evaluate_capacity(user_positions, bs_set_temp, data_rate_requirements, data_rate_capacity):
    """
    Evaluates the capacity assigned to each user based on BS coverage and available capacity.

    Parameters:
    - user_positions (np.ndarray): Array of user positions with shape (num_users, 2).
    - bs_set_temp (np.ndarray): Array of BS positions with shape (num_bs, 2).
    - data_rate_requirements (np.ndarray): Array of user data rate requirements with shape (num_users,).
    - data_rate_capacity (float): Maximum data rate capacity of each BS.

    Returns:
    - capacities (np.ndarray): Array of capacities assigned to each user with shape (num_users,).
    """
    num_users = len(user_positions)
    num_bs = len(bs_set_temp)
    capacities = np.zeros(num_users)  # Capacity assigned to each user
    bs_loads = np.zeros(num_bs)       # Load on each BS

    for user_index, user in enumerate(user_positions):
        # Calculate distances to all BSs
        distances = np.linalg.norm(bs_set_temp - user, axis=1)
        in_range = distances < 1000  # Coverage radius

        if np.any(in_range):
            # BSs that cover the user
            bs_indices = np.where(in_range)[0]

            # Calculate remaining capacities of these BSs
            remaining_capacities = data_rate_capacity - bs_loads[bs_indices]

            # Find relative indices within remaining_capacities that can accommodate the user
            feasible_bs_relative = np.where(remaining_capacities >= data_rate_requirements[user_index])[0]

            if feasible_bs_relative.size > 0:
                # Select the BS with the most remaining capacity
                selected_bs_relative = feasible_bs_relative[np.argmax(remaining_capacities[feasible_bs_relative])]
                selected_bs = bs_indices[selected_bs_relative]

                # Assign the user's data rate to this BS
                capacities[user_index] = data_rate_requirements[user_index]
                bs_loads[selected_bs] += data_rate_requirements[user_index]
            else:
                # No BS can accommodate the user
                capacities[user_index] = 0
        else:
            # User is out of range of all BSs
            capacities[user_index] = 0

    return capacities

def process_loss(user_positions, data_rate_requirements, bs_positions, capacity_threshold, data_rate_capacity):
    """
    Simulates BS loss and identifies unserviced users.

    Parameters:
    - user_positions (np.ndarray): Array of user positions with shape (num_users, 2).
    - data_rate_requirements (np.ndarray): Array of user data rate requirements with shape (num_users,).
    - bs_positions (np.ndarray): Array of all BS positions with shape (num_bs, 2).
    - capacity_threshold (float): Threshold to determine if a user is unserviced.
    - data_rate_capacity (float): Maximum data rate capacity of each BS.

    Returns:
    - unserviced_users (list of np.ndarray): List containing arrays of unserviced user indices for each loss scenario.
    """
    loss_set = [i * 0.1 for i in range(1, 10)]  # Simulate 10% to 90% loss
    unserviced_users = []

    for loss in loss_set:
        num_bs = round(len(bs_positions) * (1 - loss))  # Remaining BSs after loss
        if num_bs == 0:
            bs_set_temp = np.empty((0, 2))  # No BSs remaining
        else:
            bs_set_temp = bs_positions[:num_bs]

        # Evaluate capacity with the remaining BSs
        capacity = evaluate_capacity(user_positions, bs_set_temp, data_rate_requirements, data_rate_capacity)

        # Identify users who are not fully serviced
        unserviced = np.where(capacity < data_rate_requirements)[0]  # Indices of unserviced users
        unserviced_users.append(unserviced)

    return unserviced_users

def evaluate_fitness(uav_positions, uav_radii, uav_altitudes, user_positions, data_rate_requirements, 
                    env_params, data_rate_capacity):
    """
    Evaluates the fitness of a UAV-BS configuration.

    Parameters:
    - uav_positions (np.ndarray): Array of UAV positions with shape (num_uavs, 2).
    - uav_radii (np.ndarray): Array of UAV coverage radii with shape (num_uavs,).
    - uav_altitudes (np.ndarray): Array of UAV altitudes with shape (num_uavs,).
    - user_positions (np.ndarray): Array of user positions with shape (num_users, 2).
    - data_rate_requirements (np.ndarray): Array of user data rate requirements with shape (num_users,).
    - env_params (tuple): Environmental parameters for path loss calculation.
    - data_rate_capacity (float): Maximum data rate capacity of each BS.

    Returns:
    - fitness_score (float): The computed fitness score.
    - covered_users (np.ndarray): Boolean array indicating covered users.
    - covered_data_rates (np.ndarray): Array of data rates for covered users.
    """
    covered_users = np.zeros(len(user_positions), dtype=bool)
    covered_data_rates = np.zeros(len(user_positions))
    uav_loads = np.zeros(NUM_UAVS)

    for uav_index in range(NUM_UAVS):
        distances = np.linalg.norm(user_positions - uav_positions[uav_index], axis=1)
        in_coverage = (distances <= uav_radii[uav_index])

        for user_index in np.where(in_coverage)[0]:
            if not covered_users[user_index]:
                path_loss = calculate_path_loss(distances[user_index], uav_altitudes[uav_index], env_params)
                if path_loss <= 100:  # Adjusted threshold
                    if uav_loads[uav_index] + data_rate_requirements[user_index] <= data_rate_capacity:
                        covered_users[user_index] = True
                        covered_data_rates[user_index] = data_rate_requirements[user_index]
                        uav_loads[uav_index] += data_rate_requirements[user_index]

    # Simulate BS loss and identify unserviced users
    unserviced_users = process_loss(user_positions, data_rate_requirements, 
                                    uav_positions, 
                                    capacity_threshold=100, 
                                    data_rate_capacity=data_rate_capacity)

    # Penalize for unserviced users due to BS loss
    penalty = sum([len(users) for users in unserviced_users])
    fitness_score = np.sum(covered_users) - penalty  # Higher coverage, lower penalty

    return fitness_score, covered_users, covered_data_rates

def crossover(parent1, parent2, crossover_rate):
    """
    Performs crossover between two parent chromosomes to produce two offspring.

    Parameters:
    - parent1 (dict): First parent chromosome.
    - parent2 (dict): Second parent chromosome.
    - crossover_rate (float): Probability of performing crossover.

    Returns:
    - child1 (dict): First offspring chromosome.
    - child2 (dict): Second offspring chromosome.
    """
    child1 = {
        'uav_positions': parent1['uav_positions'].copy(),
        'uav_radii': parent1['uav_radii'].copy(),
        'uav_altitudes': parent1['uav_altitudes'].copy()
    }
    child2 = {
        'uav_positions': parent2['uav_positions'].copy(),
        'uav_radii': parent2['uav_radii'].copy(),
        'uav_altitudes': parent2['uav_altitudes'].copy()
    }

    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, NUM_UAVS)
        # Crossover UAV Positions
        child1['uav_positions'][crossover_point:], child2['uav_positions'][crossover_point:] = \
            parent2['uav_positions'][crossover_point:].copy(), parent1['uav_positions'][crossover_point:].copy()
        # Crossover UAV Radii
        child1['uav_radii'][crossover_point:], child2['uav_radii'][crossover_point:] = \
            parent2['uav_radii'][crossover_point:].copy(), parent1['uav_radii'][crossover_point:].copy()
        # Crossover UAV Altitudes
        child1['uav_altitudes'][crossover_point:], child2['uav_altitudes'][crossover_point:] = \
            parent2['uav_altitudes'][crossover_point:].copy(), parent1['uav_altitudes'][crossover_point:].copy()

    return child1, child2

def mutate(chromosome, mutation_rate):
    """
    Mutates a chromosome by randomly altering UAV positions, radii, and altitudes.

    Parameters:
    - chromosome (dict): Chromosome to mutate.
    - mutation_rate (float): Probability of mutation for each gene.

    Returns:
    - chromosome (dict): Mutated chromosome.
    """
    for uav_index in range(NUM_UAVS):
        if np.random.rand() < mutation_rate:
            chromosome['uav_positions'][uav_index] = np.random.rand(2) * 5000
        if np.random.rand() < mutation_rate:
            chromosome['uav_radii'][uav_index] = np.random.rand() * 1000
        if np.random.rand() < mutation_rate:
            chromosome['uav_altitudes'][uav_index] = np.random.uniform(50, 500)
    return chromosome

# ------------------------------
# Genetic Algorithm
# ------------------------------

def genetic_algorithm_with_loss(user_positions, data_rate_requirements, env_params, mutation_rate, crossover_rate, 
                                mobility_model, speed, group_refs=None, user_displacement=None):
    """
    Performs GA optimization considering BS loss scenarios.

    Parameters:
    - user_positions (np.ndarray): Array of user positions with shape (num_users, 2).
    - data_rate_requirements (np.ndarray): Array of user data rate requirements with shape (num_users,).
    - env_params (tuple): Environmental parameters for path loss calculation.
    - mutation_rate (float): Mutation rate for GA.
    - crossover_rate (float): Crossover rate for GA.
    - mobility_model (str): Mobility model ('random_walk' or 'rpgm').
    - speed (float): User movement speed.
    - group_refs (np.ndarray or None): Group reference points for RPGM.
    - user_displacement (np.ndarray or None): User displacement vectors for RPGM.

    Returns:
    - max_fitness (float): Maximum fitness score achieved.
    - best_fitness_scores (list): List of best fitness scores over iterations.
    - best_solution (dict): Best chromosome found.
    - best_covered_users (np.ndarray): Boolean array indicating covered users.
    - best_covered_data_rates (np.ndarray): Array of data rates for covered users.
    """
    # Initialize Population
    population = [
        {
            'uav_positions': np.random.rand(NUM_UAVS, 2) * 5000,
            'uav_radii': np.random.rand(NUM_UAVS) * 1000,
            'uav_altitudes': np.random.uniform(50, 500, NUM_UAVS)
        }
        for _ in range(POPULATION_SIZE)
    ]

    best_fitness_scores = []

    for iteration in range(NUM_ITERATIONS):
        if (iteration + 1) % 1000 == 0:
            print(f"GA Iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Update user positions based on mobility model
        user_positions, group_refs = update_user_positions(user_positions, 
                                                           mobility_model=mobility_model, 
                                                           speed=speed, 
                                                           group_refs=group_refs, 
                                                           user_displacement=user_displacement)

        # Evaluate fitness for each chromosome
        fitness_scores = []
        coverage_info = []
        for chromosome in population:
            fitness, covered_users, covered_data_rates = evaluate_fitness(
                chromosome['uav_positions'],
                chromosome['uav_radii'],
                chromosome['uav_altitudes'],
                user_positions,
                data_rate_requirements,
                env_params,
                DATA_RATE_CAPACITY
            )
            fitness_scores.append(fitness)
            coverage_info.append((covered_users.copy(), covered_data_rates.copy()))

        fitness_scores = np.array(fitness_scores)
        best_fitness = np.max(fitness_scores)
        best_fitness_scores.append(best_fitness)

        # Elitism: Select top individuals to carry over to the next generation
        elites_indices = fitness_scores.argsort()[-ELITE_SIZE:]
        elites = [population[i] for i in elites_indices]

        # Selection: Roulette Wheel Selection based on fitness scores
        total_fitness = np.sum(fitness_scores)
        if total_fitness == 0:
            selection_prob = np.ones(POPULATION_SIZE) / POPULATION_SIZE
        else:
            selection_prob = fitness_scores / total_fitness

        # Create New Population
        new_population = elites.copy()
        while len(new_population) < POPULATION_SIZE:
            parents_indices = np.random.choice(range(POPULATION_SIZE), size=2, p=selection_prob)
            parent1 = population[parents_indices[0]]
            parent2 = population[parents_indices[1]]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    # Final Evaluation to get the best solution
    final_fitness_scores = []
    final_coverage_info = []
    for chromosome in population:
        fitness, covered_users, covered_data_rates = evaluate_fitness(
            chromosome['uav_positions'],
            chromosome['uav_radii'],
            chromosome['uav_altitudes'],
            user_positions,
            data_rate_requirements,
            env_params,
            DATA_RATE_CAPACITY
        )
        final_fitness_scores.append(fitness)
        final_coverage_info.append((covered_users.copy(), covered_data_rates.copy()))

    final_fitness_scores = np.array(final_fitness_scores)
    max_fitness = np.max(final_fitness_scores)
    best_solution = population[np.argmax(final_fitness_scores)]
    best_coverage_users, best_coverage_data_rates = final_coverage_info[np.argmax(final_fitness_scores)]

    return max_fitness, best_fitness_scores, best_solution, best_coverage_users, best_coverage_data_rates

# ------------------------------
# Visualization Functions
# ------------------------------

def plot_coverage_map(user_positions, data_rate_requirements, uav_positions, uav_radii, coverage_info, env_name, num_users):
    """
    Plots the coverage map showing user distribution, UAV-BS positions, and coverage areas.

    Parameters:
    - user_positions (np.ndarray): Array of user positions with shape (num_users, 2).
    - data_rate_requirements (np.ndarray): Array of user data rate requirements with shape (num_users,).
    - uav_positions (np.ndarray): Array of UAV positions with shape (num_uavs, 2).
    - uav_radii (np.ndarray): Array of UAV coverage radii with shape (num_uavs,).
    - coverage_info (np.ndarray): Boolean array indicating covered users.
    - env_name (str): Name of the environment scenario.
    - num_users (int): Number of UEs.
    """
    plt.figure(figsize=(10, 10))

    # Define color mapping for different data rates
    color_map = {5e6: 'red', 2e6: 'green', 1e6: 'blue'}
    for rate in DATA_RATE_CHOICES:
        indices = np.where(data_rate_requirements == rate)[0]
        plt.scatter(user_positions[indices, 0], user_positions[indices, 1], 
                    c=color_map[rate], label=f'UE {int(rate/1e6)} Mbps', alpha=0.6, edgecolors='w', s=50)

    # Highlight covered users with distinct markers
    covered_indices = np.where(coverage_info == True)[0]
    plt.scatter(user_positions[covered_indices, 0], user_positions[covered_indices, 1], 
                facecolors='none', edgecolors='yellow', linewidths=1.5, label='Covered Users')

    # Plot UAV-BS positions and coverage circles
    for i in range(NUM_UAVS):
        plt.scatter(uav_positions[i, 0], uav_positions[i, 1], c='black', marker='^', s=100, 
                    label='UAV-BS' if i == 0 else "")
        circle = Circle((uav_positions[i, 0], uav_positions[i, 1]), uav_radii[i], 
                        color='black', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.title(f'Coverage Map for {env_name} Environment with {num_users} UEs')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.grid(True)
    plt.show()

def plot_coverage_by_data_rate(covered_data_rates, env_name, num_users):
    """
    Plots the number of covered users categorized by their data rate requirements.

    Parameters:
    - covered_data_rates (np.ndarray): Array of data rates for covered users.
    - env_name (str): Name of the environment scenario.
    - num_users (int): Number of UEs.
    """
    # Filter out zero data rates (uncovered users)
    rates, counts = np.unique(covered_data_rates, return_counts=True)
    rates = rates[rates > 0]
    counts = counts[rates > 0]

    plt.figure(figsize=(8, 6))
    plt.bar([f'{int(rate/1e6)} Mbps' for rate in rates], counts, color=['red', 'green', 'blue'])
    plt.xlabel('Data Rate Requirements')
    plt.ylabel('Number of Covered Users')
    plt.title(f'Coverage by Data Rate in {env_name} Environment with {num_users} UEs')
    plt.grid(axis='y')
    plt.show()

# ------------------------------
# Main Simulation
# ------------------------------

def main():
    """
    Main function to run the UAV-BS coverage optimization simulation.
    """
    coverage_ratios = {env: [] for env in ENVIRONMENTS.keys()}
    coverage_by_data_rate = {env: [] for env in ENVIRONMENTS.keys()}

    mobility_model = 'rpgm'  # Change to 'random_walk' if desired

    for env_name, env_params in ENVIRONMENTS.items():
        print(f"\nSimulating for environment: {env_name}")
        total_coverage = []
        data_rate_coverage = []

        for num_users in NUM_USERS_LIST:
            print(f"  Number of UEs: {num_users}")
            user_positions, data_rate_requirements = initialize_users(num_users)

            if mobility_model == 'rpgm':
                num_groups = max(1, num_users // 4)
                group_refs = np.random.rand(num_groups, 2) * 5000
                user_displacement = (np.random.rand(num_users, 2) - 0.5) * 100  # Random displacement within ±50m
            else:
                group_refs = None
                user_displacement = None

            # Perform GA optimization
            max_coverage, fitness_over_time, best_solution, best_covered_users, best_covered_data_rates = genetic_algorithm_with_loss(
                user_positions.copy(),
                data_rate_requirements,
                env_params,
                MUTATION_RATE,
                CROSSOVER_RATE,
                mobility_model,
                speed=10,
                group_refs=group_refs,
                user_displacement=user_displacement
            )

            coverage_ratio = max_coverage / num_users
            total_coverage.append(coverage_ratio)
            coverage_ratios[env_name].append(coverage_ratio)

            # Count covered users by data rate
            coverage_counts = {
                5e6: np.sum(best_covered_data_rates == 5e6),
                2e6: np.sum(best_covered_data_rates == 2e6),
                1e6: np.sum(best_covered_data_rates == 1e6)
            }
            data_rate_coverage.append(coverage_counts)
            coverage_by_data_rate[env_name].append(coverage_counts)

            print(f"    Coverage Ratio: {coverage_ratio * 100:.2f}%")
            print(f"    Covered Users by Data Rate: {coverage_counts}")

            if num_users == 80:
                plot_coverage_map(
                    user_positions,
                    data_rate_requirements,
                    best_solution['uav_positions'],
                    best_solution['uav_radii'],
                    best_covered_users,
                    env_name,
                    num_users
                )
                plot_coverage_by_data_rate(
                    best_covered_data_rates,
                    env_name,
                    num_users
                )

        coverage_ratios[env_name] = total_coverage
        coverage_by_data_rate[env_name] = data_rate_coverage

    # Plot Coverage Ratio vs. Number of UEs for Different Environments
    plt.figure(figsize=(10, 6))
    for env_name in ENVIRONMENTS.keys():
        plt.plot(NUM_USERS_LIST, coverage_ratios[env_name], marker='o', label=env_name)

    plt.xlabel('Number of UEs')
    plt.ylabel('Coverage Ratio')
    plt.title('Coverage Ratio vs. Number of UEs for Different Environments')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Coverage by Data Rate for Each Environment
    for env_name in ENVIRONMENTS.keys():
        plt.figure(figsize=(10, 6))
        for idx, num_users in enumerate(NUM_USERS_LIST):
            counts = coverage_by_data_rate[env_name][idx]
            plt.bar([f'{int(rate/1e6)} Mbps' for rate in DATA_RATE_CHOICES], 
                    [counts[5e6], counts[2e6], counts[1e6]], 
                    alpha=0.5, label=f'{num_users} UEs')

        plt.xlabel('Data Rate Requirements')
        plt.ylabel('Number of Covered Users')
        plt.title(f'Coverage by Data Rate in {env_name} Environment')
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
