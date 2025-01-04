from experiment.MacroBaseSet import MacroBaseSet
from experiment.UserEquipmentSet import UserEquipmentSet
from experiment.UAVset import UAVset
from experiment.graph_path import graph_path
from utilities.service_ratio import service_ratio
from network.AssignedBS import assignedBS
from network.CapacityCal import capacity
from network.ClusterCentroid import centroids
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Voronoi, voronoi_plot_2d
from utilities.PSO import PSO
from utilities.ACO import ACO
import numpy as np

class Experiment:
    '''
    Represents an experiment that simulates the interaction between UEs, BSs, and UAVs.
    Each experiment handles UE points, working BSs, and UAV deployment, including mutations and fitness evaluations.
    '''

    def __init__(self, UEset, BSset, capacity_threshold):
        '''
        Initialize the experiment with UE and BS sets, and define capacity threshold.
        '''
        self.capacity_threshold = capacity_threshold
        self.BSset = BSset
        self.UEset = UEset
        self.capacities = []
        self.unserviced = []
        self.results = []
        self.coverage_results = []
        
        # Initialize self.uavs as an empty list to avoid AttributeError in plot_power_consumption
        self.uavs = []  # This should be populated when UAVs are actually set up

        # Assign BS to each UE initially
        ue_positions = list(zip(UEset.xue, UEset.yue))
        bs_positions = list(zip(BSset.xbs, BSset.ybs))
        self.initial_assigned_BS = assignedBS(ue_positions, bs_positions)
        self.initial_capacity = capacity(ue_positions, bs_positions, self.initial_assigned_BS, capacity_threshold)
        
    # Other existing methods...

    def plot_power_consumption(self):
        '''
        Plot the power consumption of each drone over the course of the experiment.
        '''
        drone_ids = range(1, len(self.uavs) + 1)
        power_consumption = [uav.calculate_power() for uav in self.uavs]  # Assuming each UAV has a method to calculate power

        plt.figure(figsize=(8, 6))
        plt.bar(drone_ids, power_consumption, color='g')
        plt.xlabel('Drone ID')
        plt.ylabel('Power Consumption [W]')
        plt.title('Power Consumption per Drone')
        plt.grid(True)
        plt.show()

    def set_scenario(self, scenario):
        '''
        Set the environment parameters for a given scenario.
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

    # Rest of the class with other plotting methods and utilities...

        
        
    #def process_loss(self):
       # '''
       # Process the base station loss and prepare the required variables for further experiments.
       # '''
        #ue_positions = list(zip(self.UEset.xue, self.UEset.yue))
       # loss_percentages = [i * 0.1 for i in range(1, 11)]
        #total_bs = len(self.BSset.xbs)

       # for loss in loss_percentages:
          #  bs_subset_x = self.BSset.xbs[:round(total_bs * loss)]
          #  bs_subset_y = self.BSset.ybs[:round(total_bs * loss)]
          #  bs_positions_subset = list(zip(bs_subset_x, bs_subset_y))

          #  assigned_bs_subset = assignedBS(ue_positions, bs_positions_subset)
          #  capacity_subset = capacity(ue_positions, bs_positions_subset, assigned_bs_subset, self.capacity_threshold)

           # self.capacities.append(capacity_subset)
           # unserviced = [ue for ue, cap in zip(ue_positions, capacity_subset) if cap[1] == 0]
           # self.unserviced.append(unserviced)
            #print("the number of unserviced UEs: %d",unserviced)
    
    def process_loss(self):
        '''
            A process to determine BS loss and all variable sets needed for further experiments.
        '''
        ue_set = list(zip([x for x in self.UEset.xue], [y for y in self.UEset.yue]))
        loss_set = [i*0.1 for i in range(1,11)]
        tot_l = len(self.BSset.xbs)
        for loss in loss_set:
            BStempX = self.BSset.xbs[:round(tot_l*loss)]
            BStempY = self.BSset.ybs[:round(tot_l*loss)]
            bs_set_temp = list(zip([x for x in BStempX], [y for y in BStempY]))
            temp_assignedBS = assignedBS([[x,y] for x,y in zip(self.UEset.xue,self.UEset.yue)],[[x,y] for x,y in zip(BStempX,BStempY)])
            self.capacities.append(capacity(ue_set, bs_set_temp, temp_assignedBS, self.capacity_threshold))
            temp_un=[]
            for user,cap in zip(ue_set, self.capacities[-1]):
                if(cap[1] == 0):
                    temp_un.append(user)
            self.unserviced.append(temp_un)

    
    def process_uav(self, specimen_population):
        '''
        Executes a set of mutation functions and through fitness functions retrieves the best candidates.
        
        Parameters
        ----------
        specimen_population : int
            The number of best specimens that are selected to keep mutating between generations.
        '''
        self.unserviced_centroids = []
        for loss in range(10):
            unserviced_tmp = self.unserviced[loss][:]
            num_unserviced_centroids = len(unserviced_tmp)
            print(f"For loss level {loss}, the number of unserviced centroids is: {num_unserviced_centroids}")
            tmp_unserviced_centroids = centroids([x[0] for x in unserviced_tmp],[x[1] for x in unserviced_tmp])
            self.unserviced_centroids.append(tmp_unserviced_centroids)
            self.uav_set = [UAVset(0,0,i,tmp_unserviced_centroids) for i in range(3, 16, 3)]
            for n,uav_option in enumerate(self.uav_set):
                best_score = 0
                candidates = []
                fit_counter = 0
                limit = 40
                while fit_counter < limit:
                    fit_counter += 1
                    uav_option.mutate()
                    uav_option.fit(unserviced_tmp,200000,5, len(tmp_unserviced_centroids), specimen_population)
                    best = uav_option.best_uav
                    if best[0] > best_score:
                        best_score = best[0]
                        candidates.append({'specimen':best[1],'score':best_score})
                        fit_counter = 0
                        
                print('Calculated swarm of drones %d out of %d for loss %d out of %d' %((n+1) *3,len(self.uav_set) *3, loss+1, 10))
                #result = {'loss':(10 - loss)*10, 'allowed_drones':(n+1) * 3, 'candidates':candidates, 'best_score':best_score, 'best_specimen':candidates[-1]}
                # Before the error line
                if candidates:  # Check if candidates is not empty
                    result = {
                        'loss': (10 - loss) * 10,
                        'allowed_drones': (n + 1) * 3,
                        'candidates': candidates,
                        'best_score': best_score,
                        'best_specimen': candidates[-1]
                    }
                else:  # Handle empty candidates case
                    result = {
                        'loss': (10 - loss) * 10,
                        'allowed_drones': (n + 1) * 3,
                        'candidates': [],
                        'best_score': 0,
                        'best_specimen': None  # Or provide a default value
                    }
                self.results.append(result)

    def plot_coverage(self):
        '''
        Plot coverage percentage vs the number of UAVs allowed for different base station loss percentages.
        '''
        for loss_data in self.coverage_results:
            loss = self.results[-1]['loss']
            allowed_drones = [entry['allowed_drones'] for entry in loss_data]
            coverage = [entry['coverage'] for entry in loss_data]

            plt.plot(allowed_drones, coverage, label=f'Loss {loss}%')

        plt.xlabel('Number of Drones Allowed')
        plt.ylabel('Coverage Percentage')
        plt.title('Coverage Percentage vs Number of Drones Allowed')
        plt.legend()
        plt.grid(True)
        plt.show()

    def graph_path(specimen_candidates, unserviced_points, unserviced_centroids, loss, drones):
        '''
        Generate and save the visualization for UAV paths and unserviced areas using Voronoi diagrams.

        Parameters
        ----------
        specimen_candidates : list
            List of candidate drone specimens.
        unserviced_points : list
            List of unserviced UE positions.
        unserviced_centroids : list
            List of centroids for unserviced areas.
        loss : float
            Percentage of base station loss.
        drones : int
            Number of drones used in the simulation.

        Returns
        -------
        None
        '''
        fig_p = 1
        vor = Voronoi(unserviced_centroids)

        for i, drone in enumerate(specimen_candidates):
            fig = plt.figure(fig_p, figsize=(10, 8))
            ax = fig.add_subplot(111)

            # Plot Voronoi diagram
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=0.75, line_alpha=0.75,
                            point_size=1.5)

            # Plot unserviced points and centroids
            ax.plot([x[0] for x in unserviced_centroids], [x[1] for x in unserviced_centroids], 'g*', markersize=10)
            ax.plot([x[0] for x in unserviced_points], [x[1] for x in unserviced_points], 'r.', markersize=3)

            # Plot the drone paths
            path = drone['specimen'].chromosomes
            color_styles = ['y--', 'c--', 'm--', 'b--', 'k--', 'y-', 'c-', 'm-', 'b-', 'k-']
            for pa in path:
                x = [x[0] for x in pa]
                x.insert(0, x[-1])
                y = [x[1] for x in pa]
                y.insert(0, y[-1])
                ax.plot(x, y, color_styles.pop(), linewidth=1)

            plt.xlabel('Km')
            plt.ylabel('Km')
            plt.title(f'Specimen {i} with a score of {drone["score"] * 100:.2f}%, '
                      f'with a {loss}% BS loss, and {drones} drone(s) allowed')

            plt.savefig(f'figure{fig_p}.png')
            plt.gcf().set_size_inches(17, 10)
            fig_p += 1
    def plot_max_score_vs_loss(self):
        '''
        Plot the maximum score achieved in the experiment as base station loss increases.
    
        Returns
        -------
        None
        '''
        loss_percentages = [i * 10 for i in range(1, 11)]
        max_scores = [max([y['best_score'] for y in self.results if y['loss'] == loss]) for loss in loss_percentages]
    
        plt.figure(figsize=(8, 6))
        plt.plot(loss_percentages, max_scores, marker='o', color='b')
        plt.xlabel('Base Station Loss Percentage')
        plt.ylabel('Maximum Score')
        plt.title('Maximum Score vs Base Station Loss Percentage')
        plt.grid(True)
        plt.show()
        
    def plot_qos_compliance(self):
        '''
        Plot the number of UEs meeting QoS requirements at each base station loss level.
    
        Returns
        -------
        None
        '''
        # Base station loss percentages (10%, 20%, ..., 100%)
        loss_percentages = [i * 10 for i in range(1, 11)]
        qos_compliance = []
    
        # Ensure that results are available for each loss level (10%, 20%, ..., 100%)
        for loss in loss_percentages:
            # Filter results for the current loss level
            loss_data = [r for r in self.results if r['loss'] == loss]
    
            if loss_data:
                # Calculate QoS compliance for the current loss level
                compliant_ues = sum(1 for candidate in loss_data[0]['candidates'] if candidate['score'] >= self.capacity_threshold)
                qos_compliance.append(compliant_ues)
            else:
                # Append 0 if no results are found for the given loss level
                qos_compliance.append(0)
    
        # Ensure that loss_percentages and qos_compliance have the same length
        if len(loss_percentages) == len(qos_compliance):
            plt.figure(figsize=(8, 6))
            plt.plot(loss_percentages, qos_compliance, marker='o', color='b')
            plt.xlabel('Base Station Loss Percentage')
            plt.ylabel('Number of UEs Meeting QoS Requirements')
            plt.title('QoS Compliance vs Base Station Loss')
            plt.grid(True)
            plt.show()
        else:
            print("Error: Mismatch in the length of loss_percentages and qos_compliance.")


    def plot_mobility_impact(self):
        '''
        Plot the coverage ratio as UEs move over time in the experiment.
    
        Returns
        -------
        None
        '''
        iterations = range(1, len(self.results) + 1)
        coverage_ratios = [100 - candidate['best_score'] for candidate in self.results]
    
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, coverage_ratios, marker='o', color='r')
        plt.xlabel('Iterations')
        plt.ylabel('Coverage Ratio [%]')
        plt.title('Coverage Ratio Over Time with Mobile UEs')
        plt.grid(True)
        plt.show()

    def plot_coverage_vs_power(self):
        '''
        Plot the trade-off between coverage and power consumption.
    
        Returns
        -------
        None
        '''
        coverage_ratios = [(100 - candidate['best_score']) for candidate in self.results]
        power_consumption = [uav.calculate_power() for uav in self.uavs]  # Assuming each UAV has a method to calculate power
    
        plt.figure(figsize=(8, 6))
        plt.scatter(coverage_ratios, power_consumption, color='m', marker='o')
        plt.xlabel('Coverage Ratio [%]')
        plt.ylabel('Power Consumption [W]')
        plt.title('Coverage vs. Power Consumption')
        plt.grid(True)
        plt.show()
   
    def plot_scenario_performance(self, scenarios):
        '''
        Plot the UAV performance (coverage) across different environmental scenarios.
    
        Parameters
        ----------
        scenarios : list
            List of scenario names to compare (e.g., ['urban', 'suburban', 'rural']).
    
        Returns
        -------
        None
        '''
        coverage_ratios = []
        for scenario in scenarios:
            self.set_scenario(scenario)  # Assuming a method exists to switch scenarios
            self.process_loss()
            coverage = [(100 - candidate['best_score']) for candidate in self.results]
            coverage_ratios.append(np.mean(coverage))
    
        plt.figure(figsize=(8, 6))
        plt.bar(scenarios, coverage_ratios, color=['b', 'g', 'r'])
        plt.xlabel('Scenario')
        plt.ylabel('Average Coverage Ratio [%]')
        plt.title('UAV Coverage Across Different Scenarios')
        plt.grid(True)
        plt.show()
        