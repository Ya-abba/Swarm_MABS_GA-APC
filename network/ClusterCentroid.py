# -*- coding: utf-8 -*-
"""
K-means Cluster Calculator with QoS and Mobility Support

@author: Jose Matamoros (updated by major)
"""
from sklearn.cluster import AffinityPropagation
import numpy as np


def centroids(xue, yue):
    '''
    Provides the centroids for a given cluster using Affinity Propagation.

    Parameters
    ----------
    xue : [float]
        Vector with X positions for the set.
    yue : float
        Vector with Y positions for the set.

    Returns
    -------
    [float]
        A Vector with X and Y positions of the centroids.
    '''
    arr = np.array([[x, y] for x, y in zip(xue, yue)])
    clustering = AffinityPropagation(damping=0.60).fit(arr)
    centroids = clustering.cluster_centers_
    return centroids


def centroids_qos(xue, yue, qos_levels):
    '''
    Provides the centroids for a given cluster, considering QoS requirements.

    Parameters
    ----------
    xue : [float]
        Vector with X positions for the set.
    yue : float
        Vector with Y positions for the set.
    qos_levels : [float]
        A vector of QoS levels associated with each user (e.g., data rate requirements).

    Returns
    -------
    [float]
        A Vector with X and Y positions of the centroids.
    '''
    arr = np.array([[x, y, qos] for x, y, qos in zip(xue, yue, qos_levels)])
    clustering = AffinityPropagation().fit(arr)
    centroids = clustering.cluster_centers_[:, :2]  # Ignore QoS in the output centroids
    return centroids


def centroids_with_mobility(xue, yue, mobility_updates):
    '''
    Recalculates centroids as user positions change due to mobility.

    Parameters
    ----------
    xue : [float]
        Original X positions for the users.
    yue : [float]
        Original Y positions for the users.
    mobility_updates : [float]
        Updated positions of users after mobility.

    Returns
    -------
    [float]
        Updated centroids based on new positions.
    '''
    updated_xue = [x + dx for x, dx in zip(xue, mobility_updates[:, 0])]
    updated_yue = [y + dy for y, dy in zip(yue, mobility_updates[:, 1])]
    
    return centroids(updated_xue, updated_yue)


def dynamic_clustering(xue, yue, max_clusters=None):
    '''
    Dynamically cluster users based on the number of UAVs or other dynamic constraints.

    Parameters
    ----------
    xue : [float]
        Vector with X positions for the set.
    yue : float
        Vector with Y positions for the set.
    max_clusters : int or None
        Maximum number of clusters to limit computational load.

    Returns
    -------
    [float]
        A Vector with X and Y positions of the centroids.
    '''
    arr = np.array([[x, y] for x, y in zip(xue, yue)])
    clustering = AffinityPropagation(damping=0.60, max_iter=max_clusters).fit(arr)
    centroids = clustering.cluster_centers_
    return centroids


if __name__ == "__main__":
    ## Testing the functions
    import matplotlib.pyplot as plt
    import numpy as np

    # Example: Generate random user positions
    num_ues = np.random.poisson(200, 1)[0]
    xue = np.random.uniform(0, 5, num_ues)
    yue = np.random.uniform(0, 5, num_ues)

    # QoS example (assuming data rate requirements)
    qos_levels = np.random.uniform(1e6, 5e6, num_ues)

    # Simple centroid calculation
    k2 = centroids(xue, yue)

    # QoS-aware centroid calculation
    k_qos = centroids_qos(xue, yue, qos_levels)

    # Simulate mobility (random movement)
    mobility_updates = np.random.uniform(-0.5, 0.5, (num_ues, 2))
    k_mobility = centroids_with_mobility(xue, yue, mobility_updates)

    # Dynamic clustering based on UAV limits
    k_dynamic = dynamic_clustering(xue, yue, max_clusters=10)

    # Plot centroids (standard and QoS-aware)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Standard Centroids")
    plt.scatter(xue, yue, color='red', label='UEs')
    plt.scatter(k2[:, 0], k2[:, 1], color='blue', label='Centroids')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("QoS-Aware Centroids")
    plt.scatter(xue, yue, color='red', label='UEs')
    plt.scatter(k_qos[:, 0], k_qos[:, 1], color='green', label='QoS Centroids')
    plt.legend()

    plt.show()
