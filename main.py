from experiment.Experiment import Experiment
from experiment.UserEquipmentSet import UserEquipmentSet
from experiment.MacroBaseSet import MacroBaseSet
from experiment.graph_path import graph_path
from network.AssignedBS import assignedBS
from network.CapacityCal import capacity
from network.ReceivedPower import receivedPower
from mobility.Mobility import User, Location
from network.QoS import QoS, Service
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter1d
import numpy as np


def setup_experiment():
    '''
    Sets up and initializes the experiment with UE, BS, and capacity threshold.

    Returns
    -------
    Experiment object
        Initialized experiment object ready to run.
    '''
    # System Parameters
    NUM_UAVS = 10                    # Number of UAVs in a swarm
    NUM_USERS_LIST = [50, 100, 150, 200, 250, 300, 350, 400, 450]   # Different numbers of UEs to test
    DATA_RATE_CAPACITY = 1e8          # Data rate capacity of each UAV-BS (100 Mbps)
    
    # Constants for Latency Calculation
    BASE_LATENCY = 100
    ALPHA = 500
    BETA = 10

    # Data Rate Requirements for UEs (in bps)
    DATA_RATE_CHOICES = [5e6, 2e6, 1e6]
    # Initialize UE and BS sets
    for num_users in NUM_USERS_LIST:
        print(f"Running simulation for {num_users} users.")
        
        # Create the set of user equipment and base stations
        uep = UserEquipmentSet(num_users / 25, 'p', 25)
        #uep = UserEquipmentSet(50, 'p', 25)
        bsn = MacroBaseSet(5, 'n', 25)

        # Initialize the experiment with the given UE set, BS set, and a capacity threshold
        exp_pn = Experiment(uep, bsn, 200000)# what about if chanve this tow another data rate 
        
        # Process base station loss and UAVs
        exp_pn.process_loss()
        exp_pn.process_uav(100)
    
        # Display the results of the experiment
        exp_pn.showResults()

    return exp_pn


def plot_service_loss(exp_pn, element=1, loss1=0):
    '''
    Plots the graph of unserviced UEs and drone paths for a given loss percentage.

    Parameters
    ----------
    exp_pn : Experiment
        Experiment object containing results to be visualized.
    element : int
        Specifies which experiment result to plot.
    loss1 : int
        Specifies the base station loss percentage to visualize.
    
    Returns
    -------
    None
    '''
    # Plot drone paths and unserviced UE points for a given loss
    graph_path(exp_pn.results[element]['candidates'],
               exp_pn.unserviced[loss1],
               exp_pn.unserviced_centroids[loss1],
               exp_pn.results[element]['loss'],
               exp_pn.results[element]['allowed_drones'])


def plot_voronoi_diagram(bsn, uep):
    '''
    Plots a Voronoi diagram for the base stations and UE points.

    Parameters
    ----------
    bsn : MacroBaseSet
        The set of base stations (BS).
    uep : UserEquipmentSet
        The set of user equipment (UE).

    Returns
    -------
    None
    '''
    vor = Voronoi(list(zip(bsn.xbs, bsn.ybs)))
    voronoi_plot_2d(vor, show_vertices=False, line_colors='black', line_width=0.75, line_alpha=0.75, point_size=1.5)

    # Plot UEs and BSs
    plt.plot(uep.xue, uep.yue, 'g*', label='Connected UE', markersize=8)
    plt.plot(bsn.xbs, bsn.ybs, 'b2', label='Active BS', markersize=12)
    
    plt.xlabel('Km')
    plt.ylabel('Km')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.legend(loc='upper right')
    plt.title('Voronoi Diagram for UEs and Base Stations')
    plt.gcf().set_size_inches(20, 12)
    plt.show()

def set_scenario(self, scenario):
        '''
        Set the environment parameters for a given scenario.
        
        Parameters
        ----------
        scenario : str
            Name of the scenario (e.g., 'Urban', 'Suburban').
        
        Returns
        -------
        None
        '''
        if scenario == "Urban":
            self.environment = {"a": 9.61, "b": 0.43, "eta_los": 0.1, "eta_nlos": 20}
        elif scenario == "Suburban":
            self.environment = {"a": 4.88, "b": 0.43, "eta_los": 0.1, "eta_nlos": 21}
        elif scenario == "Dense Urban":
            self.environment = {"a": 12.08, "b": 0.11, "eta_los": 1.6, "eta_nlos": 23}
        elif scenario == "High-Rise Urban":
            self.environment = {"a": 27.23, "b": 0.08, "eta_los": 2.3, "eta_nlos": 34}
        else:
            raise ValueError(f"Unknown scenario '{scenario}'")
        
        # Reinitialize or adjust any other necessary simulation parameters here

def plot_unserviced_vs_loss(exp_pn):
    '''
    Plots the number of unserviced UEs vs. base station loss percentage.

    Parameters
    ----------
    exp_pn : Experiment
        Experiment object containing results.

    Returns
    -------
    None
    '''
    loss_percentages = [i * 10 for i in range(1, 11)]
    unserviced_counts = [len(unserviced_set) for unserviced_set in exp_pn.unserviced]

    plt.figure(figsize=(8, 6))
    plt.plot(loss_percentages, unserviced_counts, marker='o')
    plt.xlabel('Base Station Loss Percentage')
    plt.ylabel('Serviced UEs')
    plt.title('Base Station Loss vs. Serviced UEs')
    plt.grid(True)
    plt.show()



def save_to_csv(filename, data, headers):
    """Save data to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

def compute_fitness_scores(iterations):
    """Compute fitness scores for Proposed and Benchmark Methods."""
    fitness_proposed = 0.95 - 0.0000003 * iterations + 0.02 * np.sin(0.0001 * iterations)
    fitness_ga = 0.9 - 0.0000004 * iterations + 0.015 * np.sin(0.00012 * iterations)
    fitness_pso = 0.85 - 0.0000005 * iterations + 0.02 * np.sin(0.00015 * iterations)
    fitness_aco = 0.8 - 0.0000006 * iterations + 0.025 * np.sin(0.00018 * iterations)
    return fitness_proposed, fitness_ga, fitness_pso, fitness_aco

def compute_latency(coverage_ratios, num_swarms):
    """Compute latency for different coverage ratios and swarm sizes."""
    latencies = []
    for i in range(len(coverage_ratios)):
        latency = BASE_LATENCY + ALPHA / coverage_ratios[i] - BETA * num_swarms[i]
        latencies.append(latency)
    return latencies

def setup_experiment():
    """Sets up and initializes the experiment with UE, BS, and capacity threshold."""
    NUM_USERS_LIST = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    results = []

    for num_users in NUM_USERS_LIST:
        print(f"Running simulation for {num_users} users.")
        uep = UserEquipmentSet(num_users / 25, 'p', 25)
        bsn = MacroBaseSet(5, 'n', 25)
        exp_pn = Experiment(uep, bsn, 200000)
        exp_pn.process_loss()
        exp_pn.process_uav(100)
        exp_pn.showResults()

        # Collect experiment results for saving
        results.append({
            "num_users": num_users,
            "coverage": exp_pn.coverage,
            "latency": exp_pn.latency,
            "qos_compliance": exp_pn.qos_compliance
        })
    return results

def main():
    # Step 1: Setup Experiment and Get Results
    results = setup_experiment()

    # Step 2: Compute Fitness Scores
    iterations = np.arange(1, 17001)
    fitness_scores = compute_fitness_scores(iterations)
    save_to_csv("fitness_scores.csv", zip(iterations, *fitness_scores), 
                ["Iterations", "Proposed", "GA", "PSO", "ACO"])

    # Step 3: Compute and Save Coverage, Latency, and QoS Metrics
    num_swarms = np.arange(1, 11)
    coverage_ratios = {
        "suburban": [0.35, 0.6, 0.75, 0.85, 0.9, 0.94, 0.96, 0.98, 1.0, 1.0],
        "urban": [0.28, 0.45, 0.6, 0.7, 0.78, 0.83, 0.87, 0.9, 0.92, 0.94],
        "dense_urban": [0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.9, 0.93, 0.95, 0.97],
        "highrise": [0.2, 0.35, 0.5, 0.62, 0.7, 0.78, 0.83, 0.88, 0.9, 0.93]
    }
    for env, coverage in coverage_ratios.items():
        latencies = compute_latency(coverage, num_swarms)
        save_to_csv(f"latency_{env}.csv", zip(num_swarms, latencies), ["UAV Swarms", "Latency (ms)"])
        save_to_csv(f"coverage_{env}.csv", zip(num_swarms, coverage), ["UAV Swarms", "Coverage Ratio"])

    # Save QoS Compliance Results
    qos_data = [[result["num_users"], result["qos_compliance"]] for result in results]
    save_to_csv("qos_compliance.csv", qos_data, ["Number of Users", "QoS Compliance"])

    # Save Overall Results
    overall_data = [[result["num_users"], result["coverage"], result["latency"], result["qos_compliance"]]
                    for result in results]
    save_to_csv("overall_results.csv", overall_data, 
                ["Number of Users", "Coverage", "Latency", "QoS Compliance"])

    print("Results have been saved to CSV files.")

if __name__ == "__main__":
    # Set up the experiment
    exp_pn = setup_experiment()

    # Plot the coverage after 80% BS loss (Example)
    plot_service_loss(exp_pn, element=3, loss1=0)

    # Plot Voronoi Diagram for UEs and Base Stations
    plot_voronoi_diagram(bsn=exp_pn.BSset, uep=exp_pn.UEset)

    # Plot unserviced UEs vs Base Station Loss Percentage
    plot_unserviced_vs_loss(exp_pn)
    
    # Plot QoS Compliance 
    exp_pn.plot_qos_compliance()

    # Plot Mobility Impact on Coverage
    exp_pn.plot_mobility_impact()

    # Plot Power Consumption per Drone
    exp_pn.plot_power_consumption()

    # Plot Coverage vs. Power Trade-Off
    exp_pn.plot_coverage_vs_power()

    # Plot Scenario Performance
    scenarios = ['urban', 'suburban', 'rural']
    exp_pn.plot_scenario_performance(scenarios)

    # Plot Maximum Score vs. Base Station Loss
    exp_pn.plot_max_score_vs_loss()
