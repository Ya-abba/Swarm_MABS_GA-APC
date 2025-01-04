import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Define the MacroBaseSet and UserEquipmentSet
class MacroBaseSet:
    def __init__(self, xbs, ybs, status):
        self.xbs = xbs
        self.ybs = ybs
        self.status = status  # True for active, False for inoperative

class UserEquipmentSet:
    def __init__(self, xue, yue, connected):
        self.xue = xue
        self.yue = yue
        self.connected = connected  # True for connected, False for unconnected

# Generate base station (BS) coordinates and status
np.random.seed(42)  # For reproducibility
num_bs = 30  # Increased number of base stations
xbs = np.random.uniform(0, 25, num_bs)
ybs = np.random.uniform(0, 25, num_bs)
status = np.random.choice([True, False], num_bs, p=[0.7, 0.3])  # 70% active, 30% failed

# Generate user equipment (UE) coordinates
num_ue = 450
xue = np.random.uniform(0, 25, num_ue)
yue = np.random.uniform(0, 25, num_ue)

# Calculate UE connection status based on base station coverage
coverage_radius = 5.0  # Define a fixed coverage radius for BS
connected = np.zeros(num_ue, dtype=bool)

# Check each UE's connectivity to active base stations
for i in range(num_ue):
    for j in range(num_bs):
        distance = np.sqrt((xue[i] - xbs[j])**2 + (yue[i] - ybs[j])**2)
        if distance <= coverage_radius and status[j]:  # Connected to an active BS
            connected[i] = True
            break

# Place UAV swarm data near failed base stations
failed_bs = np.column_stack((xbs[~status], ybs[~status]))  # Coordinates of failed BS
num_swarms = len(failed_bs)  # One swarm per failed base station
swarm_leaders = failed_bs  # Swarm leaders are placed at failed BS locations
swarm_followers = np.array([
    leader + np.random.uniform(-2, 2, (2, 2))  # Two followers per leader
    for leader in swarm_leaders
])

# Create instances of the classes
bsn = MacroBaseSet(xbs, ybs, status)
uep = UserEquipmentSet(xue, yue, connected)

# Function to plot the scenario
def plot_disaster_voronoi(bsn, uep, swarm_leaders, swarm_followers, save_path="disaster_scenario.eps"):
    '''
    Plots a disaster-stricken area with Voronoi cells, UAV swarms, and coverage status.

    Parameters
    ----------
    bsn : MacroBaseSet
        The set of base stations (BS) with statuses.
    uep : UserEquipmentSet
        The set of user equipment (UE) with connection statuses.
    swarm_leaders : array
        Coordinates of UAV swarm leaders.
    swarm_followers : array
        Coordinates of UAV swarm followers.

    Returns
    -------
    None
    '''
    vor = Voronoi(list(zip(bsn.xbs, bsn.ybs)))
    voronoi_plot_2d(vor, show_vertices=False, line_colors='black', line_width=0.75, line_alpha=0.75, point_size=1.5)

    # Plot UEs
    plt.scatter(uep.xue[uep.connected], uep.yue[uep.connected], color='green', label='Connected UE', s=50, marker='*')
    plt.scatter(uep.xue[~uep.connected], uep.yue[~uep.connected], color='red', label='Disconnected UE', s=50, marker='*')

    # Plot BSs
    plt.scatter(bsn.xbs[bsn.status], bsn.ybs[bsn.status], color='blue', label='Active BS', s=100, marker='v')
    plt.scatter(bsn.xbs[~bsn.status], bsn.ybs[~bsn.status], color='black', label='Failed BS', s=100, marker='v')

    # Plot UAV swarms
    plt.scatter(swarm_leaders[:, 0], swarm_leaders[:, 1], color='orange', label='Swarm Leader', s=80, marker='o')
    for i in range(len(swarm_leaders)):
        followers = swarm_followers[i]
        plt.scatter(followers[:, 0], followers[:, 1], color='purple', label='Swarm Follower', s=50, marker='x')
        plt.plot([swarm_leaders[i, 0], followers[0, 0]], [swarm_leaders[i, 1], followers[0, 1]], 'k--', alpha=0.5)
        plt.plot([swarm_leaders[i, 0], followers[1, 0]], [swarm_leaders[i, 1], followers[1, 1]], 'k--', alpha=0.5)

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.xlabel('Km')
    plt.ylabel('Km')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.title('Disaster Scenario: 30% failed BS and UAV Swarm Support')
    plt.gcf().set_size_inches(20, 12)
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    plt.show()


# Save the plot to an EPS file
plot_disaster_voronoi(bsn, uep, swarm_leaders, swarm_followers, save_path="disaster_scenario.eps")

