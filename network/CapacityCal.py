from network.ReceivedPower import receivedPower  # Import with the updated path
from network.AssignedBS import distance, assignedUAV  # Import with the updated path
from math import log, sqrt


def db_to_w(power_db):
    '''Converts dB to watts.'''
    return 10 ** (power_db / 10)


def capacity(ue_set, bs_set, assigned_bs_list, capacity_threshold, ue_qos_requirements=None):
    '''
    Calculate each UE capacity based on the received power from all BSs, while enforcing QoS requirements.

    Parameters
    ----------
    ue_set : list
        UE set object that includes all positions.
    bs_set : list
        BS set object that includes all positions.
    assigned_bs_list : list
        BS index and power assigned in order for each UE.
    capacity_threshold : float
        The capacity threshold in bps.
    ue_qos_requirements : list, optional
        List of QoS requirements (data rate) for each UE.

    Returns
    -------
    list of lists
        A 2D list where [0] is the capacity in bps and [1] is a BOOLEAN indicating whether it's within the threshold.
    '''
    nloss = -102  # Noise floor
    bw = 1.4e6  # Bandwidth in Hz
    ue_capacity = []

    for i, (ue, a_bs) in enumerate(zip(ue_set, assigned_bs_list)):
        noise = 0
        ue_data_rate = ue_qos_requirements[i] if ue_qos_requirements else 0

        # Calculate noise from all other base stations
        for j, bs in enumerate(bs_set):
            if j != a_bs[0]:
                try:
                    noise += db_to_w(receivedPower(30, distance(ue, bs), 20)) ** 2
                except Exception as e:
                    print(f'Error in capacity calculation: {e}')
                    print(f'UE: {ue}, BS: {bs}')

        # Calculate Signal-to-Noise Ratio (SNR)
        snr = abs(db_to_w(a_bs[1]) / sqrt(noise / (len(bs_set) - 1)))
        capacity_value = bw * log(1 + snr, 10)

        # Check if UE capacity meets the threshold and QoS requirements
        if capacity_value >= capacity_threshold and a_bs[1] > nloss and (ue_data_rate == 0 or capacity_value >= ue_data_rate):
            ue_capacity.append([capacity_value, 1])
        else:
            ue_capacity.append([capacity_value, 0])

    return ue_capacity


def capacityBS_from_UE(ue_set, bs_set, assigned_ue_list, capacity_threshold):
    '''
    Calculate each UE capacity based on the received power from BSs, reverse mapping from BS to UE.

    Parameters
    ----------
    ue_set : list
        UE set object that includes all positions.
    bs_set : list
        BS set object that includes all positions.
    assigned_ue_list : list
        BS index and power assigned with every UE that it services.
    capacity_threshold : float
        The capacity threshold in bps.

    Returns
    -------
    list of lists
        A 2D list where [0] is the capacity in bps for each UE, and [1] is a BOOLEAN whether it's within the threshold.
    '''
    bw = 1.4e6  # Bandwidth in Hz
    bs_to_ue_capacity = [[] for _ in range(len(ue_set))]

    for bs, bs_pos in zip(assigned_ue_list, bs_set):
        for i, ue in enumerate(bs):
            noise = 1 if len(bs) == 1 else 0  # Avoid divide-by-zero in single UE case

            # Add noise contribution from other UEs served by the same BS
            for j, ue_extra in enumerate(bs):
                if i != j:
                    noise += db_to_w(receivedPower(-10, distance(ue_extra, bs_pos), 20)) ** 2

            # Calculate Signal-to-Noise Ratio (SNR)
            snr = abs(db_to_w(ue[1]) / sqrt(noise / (len(bs) - 1))) if len(bs) != 1 else abs(db_to_w(ue[1]) / (-112))

            # Calculate capacity
            capacity_value = bw * log(1 + snr, 10)

            # Check if capacity meets the threshold
            if capacity_value >= capacity_threshold and ue[1] > -102:
                bs_to_ue_capacity[ue[0]] = [capacity_value, 1]
            else:
                bs_to_ue_capacity[ue[0]] = [capacity_value, 0]

    return bs_to_ue_capacity


if __name__ == "__main__":
    # Example Test Cases
    # Assuming UE positions and BS positions are provided and assigned using AssignedBS module

    # Example UE and BS positions
    ue_set = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Positions of UEs
    bs_set = [(0, 1), (2, 2), (3, 4)]  # Positions of BSs

    # Example Assigned BS List
    assigned_bs_list = [(0, -70), (1, -80), (2, -90), (0, -85)]  # Example BS assignment and power for each UE

    # Example QoS Data Rate Requirements for UEs (in bps)
    ue_qos_requirements = [500000, 1000000, 1500000, 2000000]

    # Capacity Threshold (in bps)
    capacity_threshold = 500000

    # Calculate Capacity
    print("Capacity Calculation with QoS Requirements:")
    result = capacity(ue_set, bs_set, assigned_bs_list, capacity_threshold, ue_qos_requirements)
    for r in result:
        print(f"Capacity: {r[0]} bps, Meets Threshold: {r[1]}")
