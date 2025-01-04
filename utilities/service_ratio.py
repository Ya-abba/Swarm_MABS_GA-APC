import numpy as np
from network.AssignedBS import distance, assignedUAV
from network.CapacityCal import capacity

def service_ratio(chromosome_set, unserviced_set, threshold):
    '''
    Calculate the ratio of unserviced user equipment (UE) after UAV paths attempt to cover them.

    Parameters
    ----------
    chromosome_set : list
        Set of UAV paths (chromosomes).
    unserviced_set : list
        List of unserviced UEs (user equipment positions).
    threshold : float
        Capacity threshold for service.

    Returns
    -------
    float
        Ratio of unserviced UEs remaining after UAV path coverage.
    '''
    unserviced_num = len(unserviced_set)
    diameter = 3
    points = []

    # Collect points along UAV paths
    for path in chromosome_set:
        path = np.array(path + [path[0]])  # Close the loop
        for i in range(len(path) - 1):
            segment = np.linspace(path[i], path[i + 1], num=max(2, int(distance(path[i], path[i + 1]) / diameter)))
            points.extend(segment)

    # Calculate capacities for all UEs with respect to UAV paths
    capacities = capacity(unserviced_set, points, assignedUAV(unserviced_set, points), threshold)

    # Identify which UEs are covered
    serviced_indices = {i for i, cap in enumerate(capacities) if cap[1] > 0}

    # Calculate and return the service ratio
    return 1 - len(serviced_indices) / unserviced_num


def service_ratio_qos(chromosome_set, unserviced_set, threshold, uav_capacity, ue_qos_requirements):
    '''
    Calculate the service ratio considering QoS, UAV capacity, and power consumption.

    Parameters
    ----------
    chromosome_set : list
        Set of UAV paths (chromosomes).
    unserviced_set : list
        Set of unserviced UEs.
    threshold : float
        Capacity threshold for service.
    uav_capacity : float
        Total capacity of the UAV-BS.
    ue_qos_requirements : list
        List of data rate requirements for each UE.

    Returns
    -------
    float
        Service ratio adjusted for QoS and capacity.
    '''
    unserviced_num = len(unserviced_set)
    diameter = 3
    points = []

    # Collect points along UAV paths
    for path in chromosome_set:
        path = np.array(path + [path[0]])  # Close the loop
        for i in range(len(path) - 1):
            segment = np.linspace(path[i], path[i + 1], num=max(2, int(distance(path[i], path[i + 1]) / diameter)))
            points.extend(segment)

    # Calculate capacities for all UEs with respect to UAV paths
    capacities = capacity(unserviced_set, points, assignedUAV(unserviced_set, points), threshold)

    total_capacity_used = 0
    serviced_indices = set()

    for i, cap in enumerate(capacities):
        if cap[1] > 0 and total_capacity_used + ue_qos_requirements[i] <= uav_capacity:
            total_capacity_used += ue_qos_requirements[i]
            serviced_indices.add(i)

    # Calculate and return the QoS-adjusted service ratio
    return 1 - len(serviced_indices) / unserviced_num


# Testing and validation
if __name__ == "__main__":
    # Example test data
    chromosome_set = [
        [[0, 0], [1, 1], [2, 2]],  # Chromosome 1
        [[0, 2], [1, 1], [2, 0]],  # Chromosome 2
    ]
    unserviced_set = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    threshold = 1.0
    uav_capacity = 1e8  # Updated capacity to match 100 Mbps (100,000,000 bps)
    ue_qos_requirements = [5e6, 2e6, 1e6, 5e6, 2e6, 1e6]  # QoS data rate for each UE

    # Test service_ratio
    print(f'Service ratio: {service_ratio(chromosome_set, unserviced_set.copy(), threshold)}')

    # Test service_ratio_qos
    print(f'Service ratio with QoS: {service_ratio_qos(chromosome_set, unserviced_set.copy(), threshold, uav_capacity, ue_qos_requirements)}')
