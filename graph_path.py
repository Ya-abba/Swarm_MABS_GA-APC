import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_voronoi_diagram(ax, vor, unserviced_centroids, unserviced_points):
    '''
    Plot the Voronoi diagram for unserviced points and centroids.

    Parameters
    ----------
    ax : Axes object
        The matplotlib axes object where the Voronoi diagram will be plotted.
    vor : Voronoi object
        The Voronoi diagram object to be plotted.
    unserviced_centroids : list
        The centroids of the unserviced points.
    unserviced_points : list
        The unserviced points (UEs) that have not been covered.

    Returns
    -------
    None
    '''
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=0.75, line_alpha=0.75,
                    point_size=1.5)

    # Plot unserviced centroids as green stars and unserviced points as red dots
    ax.plot([x[0] for x in unserviced_centroids], [x[1] for x in unserviced_centroids], 'g*', markersize=10,
            label='Unserviced Centroids')
    ax.plot([x[0] for x in unserviced_points], [x[1] for x in unserviced_points], 'r.', markersize=3,
            label='Unserviced Points')


def plot_drone_paths(ax, drone, color_styles, qos_level=None):
    '''
    Plot the paths taken by each drone with an option for QoS-based coloring.

    Parameters
    ----------
    ax : Axes object
        The matplotlib axes object where the drone paths will be plotted.
    drone : dict
        A dictionary containing drone path and score information.
    color_styles : list
        A list of color styles for plotting the paths.
    qos_level : str, optional
        QoS level that could affect the line color or style.

    Returns
    -------
    None
    '''
    path = drone['specimen'].chromosomes
    for pa in path:
        x = [x[0] for x in pa]
        x.insert(0, x[-1])  # Close the path by connecting the last point to the first
        y = [x[1] for x in pa]
        y.insert(0, y[-1])

        # Adjust line style based on QoS level if provided
        if qos_level == 'high':
            ax.plot(x, y, 'g-', linewidth=2, label=f'Drone Path {drone["score"]:.2f} QoS: High')
        elif qos_level == 'medium':
            ax.plot(x, y, 'b-', linewidth=1.5, label=f'Drone Path {drone["score"]:.2f} QoS: Medium')
        elif qos_level == 'low':
            ax.plot(x, y, 'r--', linewidth=1, label=f'Drone Path {drone["score"]:.2f} QoS: Low')
        else:
            ax.plot(x, y, color_styles.pop(), linewidth=1, label=f'Drone Path {drone["score"]:.2f}')


def graph_path(specimen_candidates, unserviced_points, unserviced_centroids, loss, drones, qos_levels=None):
    '''
    Visualize drone paths and unserviced areas using Voronoi diagrams and path plotting.

    Parameters
    ----------
    specimen_candidates : list
        A list of candidate drone specimens with paths (chromosomes) to be visualized.
    unserviced_points : list
        List of unserviced user equipment (UE) positions.
    unserviced_centroids : list
        List of centroids for the unserviced areas.
    loss : float
        Percentage of base station loss.
    drones : int
        Number of drones used in the simulation.
    qos_levels : list, optional
        Optional QoS levels for distinguishing between drones (e.g., for color or style variation).

    Returns
    -------
    None
    '''
    fig_p = 1

    # Calculate Voronoi diagram for the unserviced centroids
    vor = Voronoi(unserviced_centroids)

    # Define different colors/styles for each drone path
    base_styles = ['y--', 'c--', 'm--', 'b--', 'k--', 'y-', 'c-', 'm-', 'b-', 'k-']

    # Loop over each drone in the specimen_candidates list
    for i, drone in enumerate(specimen_candidates):
        # Create a new figure for each drone's path
        fig1 = plt.figure(fig_p, figsize=(10, 8))
        ax_voronoi = fig1.add_subplot(111)

        # Plot the Voronoi diagram for unserviced areas
        plot_voronoi_diagram(ax_voronoi, vor, unserviced_centroids, unserviced_points)

        # Plot the path taken by the drone using different line styles/colors
        color_styles = base_styles.copy()

        # Use QoS levels if provided, otherwise use default coloring
        qos_level = qos_levels[i] if qos_levels and i < len(qos_levels) else None
        plot_drone_paths(ax_voronoi, drone, color_styles, qos_level=qos_level)

        # Add labels, title, and show the figure
        ax_voronoi.set_xlabel('Km')
        ax_voronoi.set_ylabel('Km')
        ax_voronoi.set_title(f'Specimen {i} with a score of {drone["score"] * 100:.2f}%, '
                             f'with a {loss}% BS loss, and {drones} drone(s) allowed')

        ax_voronoi.legend(loc='upper right')
        plt.savefig(f'figure{fig_p}.png')
        plt.gcf().set_size_inches(17, 10)
        fig_p += 1


if __name__ == "__main__":
    # Example testing for graph_path

    # Example unserviced points and centroids
    unserviced_points = [(0, 0), (1, 2), (2, 1), (3, 3)]
    unserviced_centroids = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]

    # Example specimen candidates (drones with paths)
    specimen_candidates = [
        {'specimen': {'chromosomes': [[(0, 0), (1, 1), (2, 2)], [(2, 2), (3, 3), (4, 4)]]}, 'score': 0.85},
        {'specimen': {'chromosomes': [[(1, 0), (2, 1), (3, 2)], [(3, 2), (4, 3), (5, 4)]]}, 'score': 0.9},
    ]

    # Example QoS levels for drones
    qos_levels = ['high', 'medium']

    # Test graph_path function with QoS levels
    graph_path(specimen_candidates, unserviced_points, unserviced_centroids, loss=20, drones=2, qos_levels=qos_levels)
