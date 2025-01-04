# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:58:48 2024

@author: major
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Parameters for simulation
num_iterations = 17000      # Number of iterations for GA
population_size = 100       # Population size for GA
mutation_rate = 0.01        # Mutation rate for GA
crossover_rate = 0.8        # Crossover rate for GA

num_uavs = 10               # Number of UAV-BSs
num_users_list = [80, 200, 450]  # Different numbers of UEs to test
data_rate_capacity = 1e8    # Data rate capacity of each UAV-BS (100 Mbps)

# Data rate requirements for UEs
data_rate_choices = [5e6, 2e6, 1e6]

# Environment parameters for different scenarios (a, b, ηLoS, ηNLoS)
environments = {
    "Suburban": (4.88, 0.43, 0.1, 21),
    "Urban": (9.61, 0.43, 0.1, 20),
    "Dense Urban": (12.08, 0.11, 1.6, 23),
    "High-Rise Urban": (27.23, 0.08, 2.3, 34),
}

# Initialize UE positions and data rates
def initialize_users(num_users):
    positions = np.random.rand(num_users, 2) * 5000  # UEs in 5000m x 5000m area
    data_rates = np.random.choice(data_rate_choices, num_users)
    return positions, data_rates

# Mobility Model Selection
def update_user_positions(user_positions, mobility_model='random_walk', speed=10, group_refs=None, user_displacement=None):
    """
    Update user positions based on the selected mobility model.
    """
    if mobility_model == 'random_walk':
        # Random Walk mobility
        angles = np.random.uniform(0, 2 * np.pi, len(user_positions))
        distances = np.random.uniform(0, speed, len(user_positions))
        delta_x = distances * np.cos(angles)
        delta_y = distances * np.sin(angles)
        user_positions[:, 0] += delta_x
        user_positions[:, 1] += delta_y
        # Ensure users stay within the bounds [0, 5000]
        user_positions[:, 0] = np.clip(user_positions[:, 0], 0, 5000)
        user_positions[:, 1] = np.clip(user_positions[:, 1], 0, 5000)
        
    elif mobility_model == 'rpgm':
        # RPGM Mobility
        num_groups = len(group_refs)
        # Move group reference points
        for i in range(num_groups):
            angles = np.random.uniform(0, 2 * np.pi)
            distances = np.random.uniform(0, speed)
            group_refs[i, 0] += distances * np.cos(angles)
            group_refs[i, 1] += distances * np.sin(angles)
            group_refs[i, 0] = np.clip(group_refs[i, 0], 0, 5000)
            group_refs[i, 1] = np.clip(group_refs[i, 1], 0, 5000)

        # Update user positions based on group reference and displacement
        for user_index in range(len(user_positions)):
            group_index = user_index % num_groups
            user_positions[user_index] = group_refs[group_index] + user_displacement[user_index]
            user_positions[user_index, 0] = np.clip(user_positions[user_index, 0], 0, 5000)
            user_positions[user_index, 1] = np.clip(user_positions[user_index, 1], 0, 5000)

    return user_positions, group_refs

# Calculate path loss (PL) based on environment parameters
def calculate_path_loss(distance, altitude, env_params):
    a, b, eta_LoS, eta_NLoS = env_params
    distance = np.maximum(distance, 1e-3)  # Avoid division by zero
    theta = (180 / np.pi) * np.arctan(altitude / distance)  # Elevation angle in degrees
    p_los = 1 / (1 + a * np.exp(-b * (theta - a)))
    p_nlos = 1 - p_los
    pl_los = 20 * np.log10(distance + altitude) + eta_LoS
    pl_nlos = 20 * np.log10(distance + altitude) + eta_NLoS
    return pl_los * p_los + pl_nlos * p_nlos

# Fitness function
def evaluate_fitness(uav_positions, uav_radii, uav_altitudes, user_positions, data_rate_requirements, env_params):
    covered_users = np.zeros(len(user_positions), dtype=bool)
    covered_data_rates = np.zeros(len(user_positions))  # To store data rates of covered users
    uav_loads = np.zeros(num_uavs)
    
    for uav_index in range(num_uavs):
        distances = np.linalg.norm(user_positions - uav_positions[uav_index], axis=1)
        in_coverage = (distances <= uav_radii[uav_index])
        
        for user_index in np.where(in_coverage)[0]:
            if not covered_users[user_index]:
                path_loss = calculate_path_loss(distances[user_index], uav_altitudes[uav_index], env_params)
                if path_loss <= 110 and (uav_loads[uav_index] + data_rate_requirements[user_index] <= data_rate_capacity):
                    covered_users[user_index] = True
                    covered_data_rates[user_index] = data_rate_requirements[user_index]
                    uav_loads[uav_index] += data_rate_requirements[user_index]
                    
    fitness_score = np.sum(covered_users)
    return fitness_score, covered_users, covered_data_rates

# Process BS loss and find unserviced users
def process_loss(user_positions, bs_positions, assigned_bs, capacity_threshold):
    """
    Simulate BS loss and find unserviced UEs.
    """
    unserviced = []
    for i, user in enumerate(user_positions):
        assigned_capacity = assigned_bs[i]  # Capacity assigned by the BS
        if assigned_capacity < capacity_threshold:
            unserviced.append(user)
    return np.array(unserviced)

# Perform GA optimization with BS loss processing and UAV deployment for unserviced UEs
def genetic_algorithm_with_loss(user_positions, data_rate_requirements, env_params, mutation_rate, crossover_rate, 
                                mobility_model, speed, group_refs=None, user_displacement=None, bs_positions=None):
    population = [
        {
            'uav_positions': np.random.rand(num_uavs, 2) * 5000,
            'uav_radii': np.random.rand(num_uavs) * 1000,
            'uav_altitudes': np.random.uniform(50, 500, num_uavs)
        }
        for _ in range(population_size)
    ]
    
    best_fitness_scores = []
    coverage_data_per_iteration = []
    
    # Step 1: Process BS loss (Unserviced UEs)
    bs_set_temp = list(zip([x for x in bs_positions[:, 0]], [y for y in bs_positions[:, 1]]))  # BS positions
    temp_assignedBS = assignedBS([[x, y] for x, y in zip(user_positions[:, 0], user_positions[:, 1])], bs_set_temp)

    # Simulate BS loss and get unserviced UEs
    capacity_threshold = 100  # Define your threshold
    unserviced_users = process_loss(user_positions, bs_positions, temp_assignedBS, capacity_threshold)
    
    # Step 2: Process UAV deployment for unserviced users
    for iteration in range(num_iterations):
        if (iteration + 1) % 1000 == 0:
            print(f"GA Iteration {iteration + 1}/{num_iterations}")
        
        # Update user positions for mobility
        user_positions, group_refs = update_user_positions(user_positions, 
                                                           mobility_model=mobility_model, 
                                                           speed=speed, 
                                                           group_refs=group_refs, 
                                                           user_displacement=user_displacement)
        
        # Evaluate fitness for each chromosome in the population
        fitness_scores = []
        coverage_info = []
        for chromosome in population:
            fitness, covered_users, covered_data_rates = evaluate_fitness(
                chromosome['uav_positions'],
                chromosome['uav_radii'],
                chromosome['uav_altitudes'],
                unserviced_users,  # Use unserviced users for UAV placement
                data_rate_requirements,
                env_params
            )
            fitness_scores.append(fitness)
            coverage_info.append((covered_users.copy(), covered_data_rates.copy()))
        
        fitness_scores = np.array(fitness_scores)
        best_fitness = np.max(fitness_scores)
        best_fitness_scores.append(best_fitness)
        
        # Selection: Roulette Wheel Selection based on fitness scores
        if np.sum(fitness_scores) == 0:
            selection_prob = np.ones(population_size) / population_size
        else:
            selection_prob = fitness_scores / np.sum(fitness_scores)
        
        # Create new population
        new_population = []
        while len(new_population) < population_size:
            parents_indices = np.random.choice(range(population_size), size=2, p=selection_prob)
            parent1 = population[parents_indices[0]]
            parent2 = population[parents_indices[1]]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # After GA iterations, evaluate final population to get the best solution
    final_fitness_scores = []
    final_coverage_info = []
    for chromosome in population:
        fitness, covered_users, covered_data_rates = evaluate_fitness(
            chromosome['uav_positions'],
            chromosome['uav_radii'],
            chromosome['uav_altitudes'],
            unserviced_users,  # Optimize for unserviced UEs
            data_rate_requirements,
            env_params
        )
        final_fitness_scores.append(fitness)
        final_coverage_info.append((covered_users.copy(), covered_data_rates.copy()))
    
    final_fitness_scores = np.array(final_fitness_scores)
    max_fitness = np.max(final_fitness_scores)
    best_solution = population[np.argmax(final_fitness_scores)]
    best_coverage_users, best_coverage_data_rates = final_coverage_info[np.argmax(final_fitness_scores)]
    
    return max_fitness, best_fitness_scores, best_solution, best_coverage_users, best_coverage_data_rates

# Helper functions for GA crossover and mutation
def crossover(parent1, parent2, crossover_rate):
    """
    Perform crossover between two parent chromosomes.
    """
    child1 = {'uav_positions': parent1['uav_positions'].copy(),
              'uav_radii': parent1['uav_radii'].copy(),
              'uav_altitudes': parent1['uav_altitudes'].copy()}
    child2 = {'uav_positions': parent2['uav_positions'].copy(),
              'uav_radii': parent2['uav_radii'].copy(),
              'uav_altitudes': parent2['uav_altitudes'].copy()}
    
    if np.random.rand() < crossover_rate:
        # Choose a random crossover point
        crossover_point = np.random.randint(1, num_uavs)
        # Swap positions
        child1['uav_positions'][crossover_point:], child2['uav_positions'][crossover_point:] = \
            parent2['uav_positions'][crossover_point:].copy(), parent1['uav_positions'][crossover_point:].copy()
        # Swap radii
        child1['uav_radii'][crossover_point:], child2['uav_radii'][crossover_point:] = \
            parent2['uav_radii'][crossover_point:].copy(), parent1['uav_radii'][crossover_point:].copy()
        # Swap altitudes
        child1['uav_altitudes'][crossover_point:], child2['uav_altitudes'][crossover_point:] = \
            parent2['uav_altitudes'][crossover_point:].copy(), parent1['uav_altitudes'][crossover_point:].copy()
    
    return child1, child2

def mutate(chromosome, mutation_rate):
    """
    Perform mutation on a chromosome.
    """
    for uav_index in range(num_uavs):
        if np.random.rand() < mutation_rate:
            # Mutate UAV position
            chromosome['uav_positions'][uav_index] = np.random.rand(2) * 5000
        if np.random.rand() < mutation_rate:
            # Mutate UAV radius
            chromosome['uav_radii'][uav_index] = np.random.rand() * 1000
        if np.random.rand() < mutation_rate:
            # Mutate UAV altitude
            chromosome['uav_altitudes'][uav_index] = np.random.uniform(50, 500)
    return chromosome

# Visualization functions
def plot_coverage_map(user_positions, data_rate_requirements, uav_positions, uav_radii, coverage_info, env_name, num_users):
    """
    Plot the coverage map showing users, UAVs, and coverage areas.
    Users are colored based on their data rate requirements.
    Covered users are highlighted.
    """
    plt.figure(figsize=(10, 10))
    
    # Plot all users
    # Define color map based on data rates
    color_map = {5e6: 'red', 2e6: 'green', 1e6: 'blue'}
    for rate in data_rate_choices:
        indices = np.where(data_rate_requirements == rate)[0]
        plt.scatter(user_positions[indices, 0], user_positions[indices, 1], 
                    c=color_map[rate], label=f'UE {int(rate/1e6)} Mbps', alpha=0.6, edgecolors='w', s=50)
    
    # Highlight covered users
    covered_indices = np.where(coverage_info == True)[0]
    plt.scatter(user_positions[covered_indices, 0], user_positions[covered_indices, 1], 
                facecolors='none', edgecolors='yellow', linewidths=1.5, label='Covered Users')
    
    # Plot UAV positions and coverage circles
    for i in range(num_uavs):
        plt.scatter(uav_positions[i, 0], uav_positions[i, 1], c='black', marker='^', s=100, label='UAV-BS' if i == 0 else "")
        # Plot coverage radius
        circle = Circle((uav_positions[i, 0], uav_positions[i, 1]), uav_radii[i], color='black', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.title(f'Coverage Map for {env_name} Environment with {num_users} UEs')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend(loc='upper right')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.grid(True)
    plt.show()

def plot_coverage_by_data_rate(covered_data_rates, env_name, num_users):
    """
    Plot the number of covered users for each data rate category.
    """
    rates, counts = np.unique(covered_data_rates, return_counts=True)
    # Only include covered users (data_rates > 0)
    rates = rates[rates > 0]
    counts = counts[rates > 0]
    
    plt.figure(figsize=(8, 6))
    plt.bar([f'{int(rate/1e6)} Mbps' for rate in rates], counts, color=['red', 'green', 'blue'])
    plt.xlabel('Data Rate Requirements')
    plt.ylabel('Number of Covered Users')
    plt.title(f'Coverage by Data Rate in {env_name} Environment with {num_users} UEs')
    plt.grid(axis='y')
    plt.show()

# Main simulation
def main():
    coverage_ratios = {env: [] for env in environments.keys()}
    coverage_by_data_rate = {env: [] for env in environments.keys()}
    
    # Choose mobility model: 'random_walk' or 'rpgm'
    mobility_model = 'rpgm'  # Change to 'random_walk' if desired
    
    for env_name, env_params in environments.items():
        print(f"\nSimulating for environment: {env_name}")
        total_coverage = []
        data_rate_coverage = []
        
        for num_users in num_users_list:
            print(f"  Number of UEs: {num_users}")
            # Initialize users
            user_positions, data_rate_requirements = initialize_users(num_users)
            
            # Initialize RPGM parameters if needed
            if mobility_model == 'rpgm':
                num_groups = max(1, num_users // 4)  # Example: 4 users per group
                group_refs = np.random.rand(num_groups, 2) * 5000
                user_displacement = (np.random.rand(num_users, 2) - 0.5) * 100  # Random displacement within ±50m
            else:
                group_refs = None
                user_displacement = None
            
            # Initialize BS positions (randomly placed for the sake of the simulation)
            bs_positions = np.random.rand(50, 2) * 5000
            
            # Run GA with BS loss processing
            max_coverage, fitness_over_time, best_solution, best_covered_users, best_covered_data_rates = genetic_algorithm_with_loss(
                user_positions.copy(),
                data_rate_requirements,
                env_params,
                mutation_rate,
                crossover_rate,
                mobility_model,
                speed=10,
                group_refs=group_refs,
                user_displacement=user_displacement,
                bs_positions=bs_positions
            )
            
            # Calculate coverage ratio
            coverage_ratio = max_coverage / num_users
            total_coverage.append(coverage_ratio)
            coverage_ratios[env_name].append(coverage_ratio)
            
            # Calculate coverage by data rate
            coverage_counts = {
                5e6: np.sum(best_covered_data_rates == 5e6),
                2e6: np.sum(best_covered_data_rates == 2e6),
                1e6: np.sum(best_covered_data_rates == 1e6)
            }
            data_rate_coverage.append(coverage_counts)
            coverage_by_data_rate[env_name].append(coverage_counts)
        print(f"    Coverage Ratio: {coverage_ratio * 100:.2f}%")
        print(f"    Covered Users by Data Rate: {coverage_counts}")
        
        # Plot coverage map for the first scenario (e.g., 80 UEs)
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

# Plot coverage ratio vs. number of UEs for different environments
plt.figure(figsize=(10, 6))
for env_name in environments.keys():
    plt.plot(num_users_list, coverage_ratios[env_name], marker='o', label=env_name)

plt.xlabel('Number of UEs')
plt.ylabel('Coverage Ratio')
plt.title('Coverage Ratio vs. Number of UEs for Different Environments')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot coverage by data rate for each environment and user count
for env_name in environments.keys():
    plt.figure(figsize=(10, 6))
    for idx, num_users in enumerate(num_users_list):
        counts = coverage_by_data_rate[env_name][idx]
        plt.bar([f'{int(rate/1e6)} Mbps' for rate in data_rate_choices], 
                [counts[5e6], counts[2e6], counts[1e6]], 
                alpha=0.5, label=f'{num_users} UEs')
    
    plt.xlabel('Data Rate Requirements')
    plt.ylabel('Number of Covered Users')
    plt.title(f'Coverage by Data Rate in {env_name} Environment')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
