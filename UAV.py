import numpy as np
from utilities import Mutators
from utilities import Fitness

class Child:
    '''
    Each class must be created for each child.
    Each child will have multiple chromosomes within, which will allow it to mutate.
    '''

    def __init__(self, current, centroids, drone_limit):
        '''
        Initialize a UAV child with chromosomes.

        Parameters
        ----------
        current : int
            The current iteration of the experiment.

        centroids : list
            If current == 0, it is a set of cluster centroids.
            If current > 0, it is a set of chromosomes.
        drone_limit : int
            The maximum number of drones allowed.
        '''
        self.born = current
        self.current = current
        if current == 0:
            self.chromosomes = Mutators.initial_build(centroids, drone_limit)
        else:
            self.chromosomes = centroids

    def mutate(self, current):
        '''
        Mutates the chromosomes and generates new children.

        Parameters
        ----------
        current : int
            The current iteration of the experiment.

        Returns
        -------
        list
            A list of new mutated children (new chromosome sets).
        '''
        self.current = current
        new_children = []
        new_children.extend(Mutators.mutator_in_chromo(self.chromosomes))

        if len(self.chromosomes) > 2:
            new_children.extend(Mutators.mutator_cross_chromo(self.chromosomes))

        return new_children

    def fitness(self, unserviced_set, threshold, uav_speed, communication_power_per_ue):
        '''
        Calculate the fitness of the child based on total distance, angle ratio, and intersections.

        Parameters
        ----------
        unserviced_set : list
            A set of unserviced UEs requiring service.
        threshold : float
            The capacity threshold for service.
        uav_speed : float
            Speed of the UAV in km/h.
        communication_power_per_ue : float
            Power required for communication with each UE in watts.

        Returns
        -------
        tuple
            A tuple containing total distance, angle ratio, intersections, and power consumption.
        '''
        # Call the correct function total_distance_and_power and handle the tuple (distance, power)
        total_dist, total_power = Fitness.total_distance_and_power(self.chromosomes, uav_speed, communication_power_per_ue)

        angle_ratio = Fitness.angle_ratio(self.chromosomes)
        intersections = Fitness.line_cross(self.chromosomes)

        return total_dist, angle_ratio, intersections, total_power

# Example Testing
if __name__ == "__main__":
    # Simulate centroids for the first generation of UAV paths
    centroids = [(1, 2), (3, 4), (5, 6)]
    child_uav = Child(current=0, centroids=centroids, drone_limit=5)

    # Mutate the UAV and generate new children
    new_children = child_uav.mutate(current=1)

    # Evaluate the fitness of the child UAV based on its chromosomes and serviceability
    unserviced_set = []  # Example unserviced UE list
    threshold = 1.0  # Capacity threshold for service
    uav_speed = 10  # UAV speed in km/h
    communication_power_per_ue = 5  # Communication power in watts

    fitness_values = child_uav.fitness(unserviced_set, threshold, uav_speed, communication_power_per_ue)

    print("Fitness values (distance, angle ratio, intersections, power consumption):")
    print(fitness_values)
