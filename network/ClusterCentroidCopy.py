from sklearn.cluster import AffinityPropagation
import numpy as np


def centroids(xue, yue):
    '''
    Provides the centroids for a given cluster

    Parameters
    ----------
    xue : [float]
        Vector with X positions for the set
    yue : float
        Vector with Y positions for the set

    Returns
    -------
    [float]
        A Vector with X and Y positions of the centroids.

    '''
    arr = []
    for x, y in zip(xue, yue):
        arr.append([x, y])

    arr = np.array(arr)
    clustering = AffinityPropagation(damping=0.60).fit(arr)
    k = clustering.cluster_centers_
    return (k)


def centroids_adv(xue, yue):
    '''
    Provides the centroids for a given cluster

    Parameters
    ----------
    xue : [float]
        Vector with X positions for the set
    yue : float
        Vector with Y positions for the set

    Returns
    -------
    [float]
        A Vector with X and Y positions of the centroids.

    '''
    arr = []
    for x, y in zip(xue, yue):
        arr.append([x, y])

    arr = np.array(arr)
    clustering = AffinityPropagation().fit(arr)
    k = clustering.cluster_centers_
    return (k)


if __name__ == "__main__":
    ##Code below used solely for testing
    import matplotlib.pyplot as plt
    import csv

    with open('Xue.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        xue = [float(ele) for ele in ([rr for rr in r][0])]

    with open('Yue.csv') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        yue = [float(ele) for ele in ([rr for rr in r][0])]

    numue = np.random.poisson(200, 1)
    xue = np.random.uniform(0, 5, numue)
    yue = np.random.uniform(0, 5, numue)

    k2 = centroids(xue, yue)
    k = centroids_adv(xue, yue)
    #    k2 = centroids_adv([x[0] for x in k],[x[1] for x in k])

    #    plt.plot(xue, yue, 'r.',[kk[0] for kk in k],[kk[1] for kk in k],'b^')
    #    plt.show()
    plt.plot(xue, yue, 'r.', [kk[0] for kk in k2], [kk[1] for kk in k2], 'b^')
    plt.show()
