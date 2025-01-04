import math
from numpy import dot
from numpy.linalg import norm
from network.AssignedBS import distance
from network.CapacityCal import capacity
from network.AssignedBS import assignedUAV


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    '''Return true if line segments AB and CD intersect'''
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_cross(chromosome_set):
    '''
    Calculate the number of intersections between all paths within the chromosome set.
    '''
    paths = [[path + [path[0]]] for path in chromosome_set]  # Close paths
    segments = [[[path[i], path[i + 1]] for i in range(len(path) - 1)] for path in paths]

    intersections = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            for line in segments[i]:
                for line2 in segments[j]:
                    if intersect(line[0], line[1], line2[0], line2[1]):
                        intersections += 1
    return intersections


def total_distance_and_power(chromosome_set, uav_speed, communication_power_per_ue):
    '''
    Calculate the total distance of UAV paths and power consumption.
    '''
    total_dist = 0
    total_power = 0

    for path in chromosome_set:
        path_dist = sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))
        total_dist += path_dist
        hover_time = len(path) * 10  # Example: 10 seconds per hover point
        comm_power = len(path) * communication_power_per_ue
        total_power += uav_power_consumption(hover_time, path_dist, uav_speed, comm_power)

    return total_dist, total_power


def uav_power_consumption(hovering_time, travel_distance, speed, communication_power):
    '''
    Calculate the total power consumed by a UAV during its mission.
    '''
    hover_power = hovering_time * 50  # Hovering power (50W per second)
    travel_power = travel_distance * speed * 10  # Travel power
    total_power = hover_power + travel_power + communication_power
    return total_power


def angle(a, b, c):
    '''Returns the angle ABC'''
    ab = distance(a, b)
    bc = distance(b, c)
    ca = distance(c, a)
    return math.acos((ab ** 2 + bc ** 2 - ca ** 2) / (2 * ab * bc))


def angle_ratio(chromosome_set):
    '''
    Returns a ratio of how much the shape resembles a perfect polygon.
    '''
    from math import pi
    ratios = []

    for path in chromosome_set:
        n = len(path)
        perfect_angle = (n - 2) * pi / n
        angles = sum(abs(angle(path[i - 1], path[i], path[(i + 1) % n]) - perfect_angle) for i in range(n))
        ratio = 1 - angles / (n * pi)
        ratios.append(ratio)

    return sum(ratios) / len(ratios)


def service_ratio_qos(chromosome_set, unserviced_set, threshold, uav_capacity, ue_qos_requirements):
    '''
    Calculates the service ratio considering QoS, UAV capacity, and power consumption.
    '''
    diameter = 3
    points = []

    for path in chromosome_set:
        path = path.copy() + [path[0]]  # Close the path
        for i in range(len(path) - 1):
            segment = np.linspace(path[i], path[i + 1], num=max(2, int(distance(path[i], path[i + 1]) / diameter)))
            points.extend(segment)

    capacities = capacity(unserviced_set, points, assignedUAV(unserviced_set, points), threshold)

    total_capacity_used = 0
    serviced_indices = set()
    
    for i, cap in enumerate(capacities):
        ue_data_rate = ue_qos_requirements[i]
        if cap[1] > 0 and total_capacity_used + ue_data_rate <= uav_capacity:
            total_capacity_used += ue_data_rate
            serviced_indices.add(i)

    return 1 - len(serviced_indices) / len(unserviced_set)


def qos_power_fitness(chromosome_set, unserviced_set, threshold, uav_speed, uav_capacity, ue_qos_requirements, communication_power_per_ue):
    '''
    Multi-objective fitness function that balances service ratio, power consumption, and QoS.
    '''
    service_ratio_val = service_ratio_qos(chromosome_set, unserviced_set, threshold, uav_capacity, ue_qos_requirements)
    total_dist, total_power = total_distance_and_power(chromosome_set, uav_speed, communication_power_per_ue)

    return service_ratio_val - 0.01 * total_power


def balanced_service_ratio(unserviced_set):
    '''Returns the ratio of the nearest neighbor distance between UEs for balanced service.'''
    nearest_neighbour_distances = [min(distance(ue, other) for j, other in enumerate(unserviced_set) if i != j) for i, ue in enumerate(unserviced_set)]
    return min(nearest_neighbour_distances) / max(nearest_neighbour_distances)


def path_smoothness_ratio(chromosome_set):
    '''
    Calculate the smoothness ratio of UAV paths based on angles between direction vectors.
    '''
    ratios = []

    for path in chromosome_set:
        directions = [[path[i + 1][j] - path[i][j] for j in range(2)] for i in range(len(path) - 1)]
        smoothness_scores = [math.acos(max(min(dot(directions[i], directions[i + 1]) / (norm(directions[i]) * norm(directions[i + 1])), 1), -1)) for i in range(len(directions) - 1)]
        ratios.append(1 - sum(smoothness_scores) / (math.pi * len(smoothness_scores)))

    return sum(ratios) / len(ratios)


# Testing and validation
if __name__ == "__main__":
    # Test data
    chromosome_set = [
        [[0, 0], [1, 1], [2, 2]],  # Chromosome 1
        [[0, 2], [1, 1], [2, 0]],  # Chromosome 2
    ]
    unserviced_set = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    threshold = 1.0
    uav_speed = 10  # Speed in km/h
    uav_capacity = 1e8  # Capacity in bps (100 Mbps)
    ue_qos_requirements = [5e6, 2e6, 1e6, 5e6, 2e6, 1e6]  # QoS data rate for each UE
    communication_power_per_ue = 5  # Communication power for each UE in watts

    fitness_score = qos_power_fitness(chromosome_set, unserviced_set, threshold, uav_speed, uav_capacity, ue_qos_requirements, communication_power_per_ue)
    print(f'Fitness score: {fitness_score}')
