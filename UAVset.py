from experiment.UAV import Child
import numpy as np
from operator import itemgetter
from functools import reduce

class UAVset:
    '''
    UAV set where each UAV is represented as a child object with multiple chromosomes.
    Each child can undergo mutation to generate new paths for optimization.
    '''

    def __init__(self, born, current, drone_limit, centroids):
        '''
        Initialize a UAVset with multiple UAVs (children), each with its own paths (chromosomes).

        Parameters
        ----------
        born : int
            Initial generation number.
        current : int
            Current generation number.
        drone_limit : int
            Maximum number of drones allowed.
        centroids : list
            Centroid positions to guide the UAV placement.
        '''
        self.current = 0
        self.drone_number = drone_limit
        self.uavs = [Child(current, centroids, drone_limit) for _ in range(50)]  # Create 50 UAVs

    def mutate(self):
        '''
        Generate new generations of UAVs by mutating existing children.

        Returns
        -------
        None
        '''
        self.current += 1
        new_generation = []
        for uav in self.uavs:
            new_generation.extend(uav.mutate(self.current))  # Add mutated UAVs to the new generation
        for uav in new_generation:
            self.uavs.append(Child(self.current, uav, self.drone_number))

    def fit(self, unservice_set, threshold, drone_limit, k, sample):
        '''
        Evaluate the fitness of UAV paths based on distance, intersection, and service coverage.

        Parameters
        ----------
        unservice_set : list
            Set of unserviced UEs.
        threshold : float
            Capacity threshold for service.
        drone_limit : int
            Maximum number of drones allowed.
        k : int
            Number of clusters.
        sample : int
            Number of best UAVs to retain for the next generation.

        Returns
        -------
        None
        '''
        distances, angle_ratios, intersections_set = [], [], []
        uav_speed = 10  # UAV speed in km/h (example value, adjust as needed)
        communication_power_per_ue = 5  # Communication power per UE in watts (example value, adjust as needed)

        for uav in self.uavs:
            # Call fitness and pass the required uav_speed and communication_power_per_ue arguments
            distance, angle_ratio, intersections, total_power = uav.fitness(unservice_set, threshold, uav_speed, communication_power_per_ue)

            distances.append(distance)
            intersections_set.append(intersections)
            angle_ratios.append(angle_ratio)

        min_intersections = min(intersections_set)
        min_distance = min(distances)

        self.uav_list = []

        # Evaluate and sort UAVs based on fitness score
        for distance_total, angle_ratio, path_intersections, uav in zip(distances, angle_ratios, intersections_set, self.uavs):
            if path_intersections != 0:
                intersection_ratio = min_intersections / path_intersections
            else:
                intersection_ratio = 1

            distance_ratio = min_distance / distance_total
            score = distance_ratio * 0.3 + angle_ratio * 0.4 + intersection_ratio * 0.3

            l1 = []
            for path in uav.chromosomes:
                for point in path:
                    l1.append(point)
            tot_k = reduce(lambda x, y: x + y, [len(x) for x in uav.chromosomes])

            # Filter UAVs that exceed drone limits or have duplicate points
            if len(uav.chromosomes) > drone_limit or tot_k != k or len(l1) != len(np.unique(l1, axis=0)):
                self.uav_list.append([0, uav])
            else:
                self.uav_list.append([score, uav])

        # Sort UAVs by score and retain the top performers
        self.uav_list = sorted(self.uav_list, key=itemgetter(0), reverse=True)
        for uav in self.uav_list[sample:]:
            del uav[1]  # Remove UAVs that did not make the cut

        self.uav_list = list(filter(lambda x: len(x) > 1, self.uav_list))
        self.uavs = [uav[1] for uav in self.uav_list]
        self.best_uav = self.uav_list[0]


if __name__ == "__main__":
    # Example test case for UAVset initialization and mutation

    # Simulate centroids and initialize UAVset
    centroids = [(1, 2), (3, 4), (5, 6)]
    uav_set = UAVset(born=0, current=0, drone_limit=3, centroids=centroids)

    # Perform mutation
    uav_set.mutate()

    # Evaluate fitness and display best UAV
    uav_set.fit(unservice_set=[], threshold=1.0, drone_limit=3, k=3, sample=5)
    print("Best UAV after fitness evaluation:")
    print(uav_set.best_uav)
