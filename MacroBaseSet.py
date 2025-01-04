import numpy as np


class MacroBaseSet:
    '''
    Defines a set of positions for macro base stations (BS) within an experiment.
    The positions are generated based on a specific density, distribution, and area size.
    '''

    def __init__(self, density, distribution, area, qos_levels=None):
        '''
        Initializes the Macro Base Station set with positions based on the given distribution.

        Parameters
        ----------
        density : float
            The density of base stations per unit area.
        distribution : str
            The type of distribution to generate BS locations ('p', 'n', 'u').
        area : float
            The area in which BSs are distributed.
        qos_levels : list, optional
            Optional QoS levels (e.g., data rate requirements) for each BS.

        Returns
        -------
        None
        '''
        from math import sqrt
        quantity = density * area

        # Select the appropriate distribution to generate base station positions
        if distribution == 'p':
            numbs = np.random.poisson(quantity, 1)  # Poisson distribution
        elif distribution == 'n':
            numbs = np.int32(np.abs(np.round(np.random.normal(quantity, 10, 1))))  # Normal distribution
        elif distribution == 'u':
            numbs = np.int32(np.abs(np.round(np.random.uniform(1, quantity, 1))))  # Uniform distribution
        else:
            raise ValueError("Unsupported distribution type. Choose 'p', 'n', or 'u'.")

        # Generate base station positions
        self.xbs = np.random.uniform(0, area, numbs)
        self.ybs = np.random.uniform(0, area, numbs)

        # Assign QoS levels if provided (e.g., data rate requirements)
        if qos_levels is not None:
            self.qos_bs = np.random.choice(qos_levels, size=len(self.xbs))
        else:
            self.qos_bs = ['default'] * len(self.xbs)  # Default QoS if none provided

    def plot_bs(self):
        '''
        Visualize the base stations and their positions.

        Returns
        -------
        None
        '''
        import matplotlib.pyplot as plt

        plt.scatter(self.xbs, self.ybs, c='blue', marker='o', label='Base Stations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Macro Base Station Positions')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example testing for initializing MacroBaseSet and plotting the BS positions

    # Initialize a MacroBaseSet with density, distribution, and area
    macro_bs_set = MacroBaseSet(density=5, distribution='p', area=100)

    # Plot the base station positions
    macro_bs_set.plot_bs()

    # Example with QoS levels
    qos_levels = ['high', 'medium', 'low']
    macro_bs_set_qos = MacroBaseSet(density=5, distribution='n', area=100, qos_levels=qos_levels)

    # Print QoS levels for base stations
    print("QoS levels for base stations:")
    print(macro_bs_set_qos.qos_bs)

    # Plot the base station positions with QoS levels
    macro_bs_set_qos.plot_bs()
