# Swarm UAV Path Planning and Coverage Optimization

This repository implements a state-of-the-art Swarm UAV path-planning algorithm designed for Disaster Response Networks (DRNs). By combining Genetic Algorithms (GA) with clustering techniques, this solution optimizes UAV deployments to maximize network coverage, ensure Quality of Service (QoS), and adapt to varying environmental conditions.

## Features

- **Coverage Optimization**: Efficiently distributes UAVs to maximize communication coverage in disaster-struck areas.
- **Genetic Algorithm (GA)**: Employs an evolutionary process to determine optimal UAV paths.
- **Mobility Models**: Simulates Random Waypoint Mobility (RWPM) and Reference Point Group Mobility (RPGM) for dynamic UE distributions.
- **QoS Compliance**: Monitors and improves service quality metrics for affected users.
- **Benchmark Comparisons**: Includes detailed comparisons with Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and GA-based benchmarks.

## Repository Contents

- `experiment/`: Scripts for setting up experiments, processing results, and analyzing performance.
- `network/`: Core modules for base station assignment, capacity calculation, and QoS evaluation.
- `mobility/`: Mobility models for simulating dynamic user movements.
- `utilities/`: Helper functions for clustering, visualization, and algorithm execution.
- `results/`: Plots and data from simulations across various scenarios.

## Prerequisites

- Python 3.8+
- Libraries: `numpy`, `matplotlib`, `scipy`, `sklearn`
- MATLAB (optional for additional benchmarking and visualization)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/swarm-uav-path-planning.git
   cd swarm-uav-path-planning
