from math import sqrt

from network.ReceivedPower import receivedPower


def distance(a, b):
    '''
    Calculate the 2D distance between points a and b.

    Parameters
    ----------
    a : [float,float]
        A set of points (x,y).

    b : [float,float]
        A set of points (x,y).

    Returns
    -------
    float
        Distance between a and b.
    '''
    d = sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return max(d, 0.001)  # Avoid returning values too close to zero.


def distanceUAV(a, b, altitude_difference=0.05):
    '''
    Calculate the 3D distance between a UE and an Aerial Base Station (UAV-BS).

    Parameters
    ----------
    a : [float,float]
        A set of points (x,y) for the UE.

    b : [float,float]
        A set of points (x,y) for the UAV-BS.

    altitude_difference : float
        The altitude difference between the UE and UAV-BS (default is 0.05).

    Returns
    -------
    float
        3D Distance between the UE and UAV-BS.
    '''
    c = ((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + (altitude_difference ** 2)
    return sqrt(c)


def assignedBS(ue_set, bs_set, transmit_power=30):
    '''
    Assigns the best BS to each UE based on received power.

    Parameters
    ----------
    ue_set : list
        List of UE positions (x, y).
    bs_set : list
        List of BS positions (x, y).
    transmit_power : float, optional
        Transmit power in dB (default is 30 dB).

    Returns
    -------
    list
        List where each entry is [best BS index, received power] for each UE.
    '''
    assigned = []
    for ue in ue_set:
        best_bs_idx = 0
        max_received_power = -1000
        for i, bs in enumerate(bs_set):
            try:
                temp_power = receivedPower(transmit_power, distance(ue, bs), 20)
                if temp_power > max_received_power:
                    max_received_power = temp_power
                    best_bs_idx = i
            except Exception as e:
                print(f"Error in calculating power for UE {ue} and BS {bs}: {e}")
        assigned.append([best_bs_idx, max_received_power])
    return assigned


def assignedUAV(ue_set, bs_set, transmit_power=20):
    '''
    Assign the best UAV-BS to each UE based on received power.

    Parameters
    ----------
    ue_set : list
        List of UE positions (x, y).
    bs_set : list
        List of UAV-BS positions (x, y).
    transmit_power : float, optional
        Transmit power in dB (default is 20 dB).

    Returns
    -------
    list
        List where each entry is [best UAV-BS index, received power] for each UE.
    '''
    assigned = []
    for ue in ue_set:
        best_uav_idx = 0
        max_received_power = -1000
        for i, bs in enumerate(bs_set):
            try:
                temp_power = receivedPower(transmit_power, distanceUAV(ue, bs), 20)
                if temp_power > max_received_power:
                    max_received_power = temp_power
                    best_uav_idx = i
            except Exception as e:
                print(f"Error in calculating power for UE {ue} and UAV-BS {bs}: {e}")
        assigned.append([best_uav_idx, max_received_power])
    return assigned


def assignedUE(ue_set, bs_set, transmit_power=30):
    '''
    Assign the best UE to each BS based on received power.

    Parameters
    ----------
    ue_set : list
        List of UE positions (x, y).
    bs_set : list
        List of BS positions (x, y).
    transmit_power : float, optional
        Transmit power in dB (default is 30 dB).

    Returns
    -------
    list
        List of lists, where each inner list contains UEs assigned to the respective BS.
    '''
    assigned = [[] for _ in range(len(bs_set))]
    for ue_idx, ue in enumerate(ue_set):
        best_bs_idx = 0
        max_received_power = -1000
        for i, bs in enumerate(bs_set):
            try:
                temp_power = receivedPower(transmit_power, distanceUAV(ue, bs), 20)
                if temp_power > max_received_power:
                    max_received_power = temp_power
                    best_bs_idx = i
            except Exception as e:
                print(f"Error in calculating power for UE {ue} and BS {bs}: {e}")
        assigned[best_bs_idx].append([ue_idx, max_received_power])
    return assigned


if __name__ == "__main__":
    # Example test cases to verify assignments

    ue_set = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Example UE positions
    bs_set = [(0, 1), (2, 2), (3, 4)]  # Example BS positions

    # Test BS assignment
    assigned_bs = assignedBS(ue_set, bs_set)
    print("Assigned BS for UEs:")
    for ue_idx, (bs_idx, power) in enumerate(assigned_bs):
        print(f"UE {ue_idx}: Best BS {bs_idx}, Received Power {power} dB")

    # Test UAV-BS assignment
    assigned_uav = assignedUAV(ue_set, bs_set)
    print("\nAssigned UAV-BS for UEs:")
    for ue_idx, (uav_idx, power) in enumerate(assigned_uav):
        print(f"UE {ue_idx}: Best UAV-BS {uav_idx}, Received Power {power} dB")

    # Test UE assignment to BSs
    assigned_ue = assignedUE(ue_set, bs_set)
    print("\nAssigned UEs for each BS:")
    for bs_idx, ue_list in enumerate(assigned_ue):
        print(f"BS {bs_idx}: UEs {ue_list}")
