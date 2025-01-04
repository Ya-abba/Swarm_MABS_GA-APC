import numpy as np
from network.ClusterCentroid import centroids, centroids_qos
from mobility.Mobility import random_waypoint_mobility, reference_point_group_mobility

class UserEquipmentSet:
    '''
    Each object will represent a set of user equipment (UE) positions and their corresponding clusters and QoS requirements.
    '''

    def __init__(self, density, distribution, area):
        '''
        Initializes the UE set with positions, data rate requirements, and clusters based on the distribution.

        Parameters
        ----------
        density : float
            The density of user equipment per unit area.
        distribution : str
            The type of distribution to generate user locations ('p', 'n', 'u').
        area : float
            The area in which users are distributed.
        '''
        quantity = density * area

        # Generate the number of UEs based on the distribution type
        if distribution == 'p':
            numue = np.random.poisson(quantity, 1)[0]  # Poisson distribution
        elif distribution == 'n':
            numue = int(abs(round(np.random.normal(quantity, 10, 1)[0])))  # Normal distribution
        elif distribution == 'u':
            numue = int(abs(round(np.random.uniform(1, quantity, 1)[0])))  # Uniform distribution
        else:
            raise ValueError("Unsupported distribution type. Choose 'p', 'n', or 'u'.")

        # Generate UE positions
        self.xue = np.random.uniform(0, area, numue)
        self.yue = np.random.uniform(0, area, numue)

        # Assign data rate requirements randomly based on given categories
        self.data_rate_categories = [5e6, 2e6, 1e6]  # Data rates: 5 Mbps, 2 Mbps, 1 Mbps
        self.data_rates = np.random.choice(self.data_rate_categories, numue, p=[0.3, 0.4, 0.3])  # Assign data rates with specified probabilities

        # Calculate clusters considering QoS
        self.clusters = centroids_qos(self.xue, self.yue, self.data_rates)

    def update_mobility(self, mobility_model='random_waypoint', speed=1.0, time_step=1.0, area_size=(100, 100)):
        '''
        Update UE positions based on the selected mobility model.

        Parameters
        ----------
        mobility_model : str
            The model of mobility ('random_waypoint', 'reference_point_group', etc.).
        speed : float
            The speed of movement for the users.
        time_step : float
            The time step for the mobility update.
        area_size : tuple
            The size of the area (width, height) for the UE movement.

        Returns
        -------
        None
        '''
        if mobility_model == 'random_waypoint':
            # Update UE positions using Random Waypoint Mobility Model
            for i in range(len(self.xue)):
                self.xue[i], self.yue[i] = random_waypoint_mobility((self.xue[i], self.yue[i]), area_size, speed, time_step)
        
        elif mobility_model == 'reference_point_group':
            # Implement Reference Point Group Mobility model
            group_center = (np.mean(self.xue), np.mean(self.yue))  # Calculate the group's reference point
            updated_positions = reference_point_group_mobility(self, group_center, speed, area_size, time_step)
            self.xue = [pos[0] for pos in updated_positions]
            self.yue = [pos[1] for pos in updated_positions]

        # Recalculate clusters after mobility update
        self.clusters = centroids_qos(self.xue, self.yue, self.data_rates)

    def get_data_rate_distribution(self):
        '''
        Returns the distribution of UEs across the three data rate categories.

        Returns
        -------
        dict
            A dictionary showing the count of UEs in each data rate category.
        '''
        unique, counts = np.unique(self.data_rates, return_counts=True)
        return dict(zip(unique, counts))

    def get_average_qos(self):
        '''
        Calculate and return the average QoS metrics (data rate) across all UEs.

        Returns
        -------
        float
            The average data rate required by the UEs.
        '''
        return np.mean(self.data_rates)

if __name__ == "__main__":
    ## Example testing for UE initialization, mobility updates, and data rate assignment

    # Initialize User Equipment Set with density and distribution
    ue_set = UserEquipmentSet(density=10, distribution='p', area=100)

    # Display initial clusters and data rates
    print("Initial Data Rate Distribution:", ue_set.get_data_rate_distribution())
    print("Initial Clusters based on QoS:")
    print(ue_set.clusters)

    # Simulate mobility (e.g., random waypoint model)
    ue_set.update_mobility(mobility_model='random_waypoint', speed=1.0, time_step=1.0, area_size=(100, 100))

    # Display updated clusters and data rates after mobility
    print("\nData Rate Distribution after Mobility Update:", ue_set.get_data_rate_distribution())
    print("Clusters after mobility update:")
    print(ue_set.clusters)
    
    # Test with reference point group mobility model
    ue_set.update_mobility(mobility_model='reference_point_group', speed=1.0, time_step=1.0, area_size=(100, 100))
    print("\nData Rate Distribution after RPGM Update:", ue_set.get_data_rate_distribution())
    print("Clusters after RPGM update:")
    print(ue_set.clusters)
