class UE:
    def __init__(self, data_rate, latency, reliability):
        '''
        Initialize a UE with QoS requirements.

        Parameters
        ----------
        data_rate : float
            UE's data rate requirement in bps.
        latency : float
            UE's latency requirement in milliseconds.
        reliability : float
            UE's reliability requirement (percentage).
        '''
        self.data_rate = data_rate  # Data rate requirement
        self.latency = latency  # Latency requirement
        self.reliability = reliability  # Reliability requirement


class UAV_BS:
    def __init__(self, capacity, latency, reliability):
        '''
        Initialize a UAV-BS with capacity and QoS characteristics.

        Parameters
        ----------
        capacity : float
            UAV-BS's data rate capacity in bps.
        latency : float
            Maximum latency the UAV-BS can handle.
        reliability : float
            Reliability of the UAV-BS as a percentage.
        '''
        self.capacity = capacity  # Data rate capacity of UAV-BS
        self.latency = latency  # Maximum allowable latency for the UAV-BS
        self.reliability = reliability  # Reliability of the UAV-BS


class Service:
    def __init__(self, data_rate, latency, reliability):
        '''
        Initialize a Service with specific QoS requirements.

        Parameters
        ----------
        data_rate : float
            Service's data rate in bps.
        latency : float
            Service's latency in milliseconds.
        reliability : float
            Service's reliability as a percentage.
        '''
        self.data_rate = data_rate
        self.latency = latency
        self.reliability = reliability


class QoS:
    '''
    QoS enforcement mechanism for UE, UAV-BS, and Services.
    '''
    UE_DATA_RATES = [5e6, 2e6, 1e6]  # UE data rates in bps
    UE_LATENCIES = [50, 100, 200]  # UE latency requirements in ms
    UE_RELIABILITIES = [99.9, 99.0, 98.0]  # UE reliability requirements as percentages
    UAV_BS_CAPACITY = 1e8  # UAV-BS data rate capacity in bps
    UAV_BS_LATENCY = 100  # Maximum allowable latency for UAV-BS (in ms)
    UAV_BS_RELIABILITY = 99.9  # UAV-BS reliability (percentage)
    CAPACITY_THRESHOLDS = [1e5, 2e5, 3e5]  # Capacity thresholds for data rate

    def __init__(self):
        self.ues = [UE(rate, lat, rel) for rate, lat, rel in zip(self.UE_DATA_RATES, self.UE_LATENCIES, self.UE_RELIABILITIES)]
        self.uav_bs = UAV_BS(self.UAV_BS_CAPACITY, self.UAV_BS_LATENCY, self.UAV_BS_RELIABILITY)

    def enforce(self, service):
        '''
        Enforce QoS requirements for a given service.

        Parameters
        ----------
        service : Service
            The service object containing data rate, latency, and reliability.

        Raises
        ------
        Exception
            If the service fails to meet the UE or UAV-BS QoS requirements.
        
        Returns
        -------
        bool
            True if QoS requirements are successfully enforced.
        '''
        # Check if service satisfies each UE's QoS requirements
        for ue in self.ues:
            if service.data_rate > ue.data_rate:
                raise Exception(f'Service data rate exceeds UE data rate capacity: {ue.data_rate} bps')
            if service.latency < ue.latency:
                raise Exception(f'Service latency {service.latency}ms is below UE requirement: {ue.latency}ms')
            if service.reliability < ue.reliability:
                raise Exception(f'Service reliability {service.reliability}% is below UE requirement: {ue.reliability}%')

        # Check if service satisfies UAV-BS QoS requirements
        if service.data_rate > self.uav_bs.capacity:
            raise Exception(f'Service data rate {service.data_rate} bps exceeds BS capacity {self.uav_bs.capacity} bps')
        if service.latency < self.uav_bs.latency:
            raise Exception(f'Service latency {service.latency}ms is below BS capability: {self.uav_bs.latency}ms')
        if service.reliability < self.uav_bs.reliability:
            raise Exception(f'Service reliability {service.reliability}% is below BS capability: {self.uav_bs.reliability}%')

        # Capacity thresholds check
        for threshold in self.CAPACITY_THRESHOLDS:
            if service.data_rate > threshold:
                raise Exception(f'Service data rate {service.data_rate} bps exceeds the predefined capacity threshold: {threshold} bps')

        # If all QoS checks pass, return True
        return True


# Testing the QoS enforcement with various services
if __name__ == "__main__":
    # Create a service with specific data rate, latency, and reliability
    service1 = Service(3e6, 60, 99.5)  # Service with data rate 3 Mbps, latency 60 ms, reliability 99.5%
    service2 = Service(10e6, 40, 99.9)  # Service exceeding UE data rate

    qos_system = QoS()

    try:
        print("Service 1: ", qos_system.enforce(service1))  # This should pass
    except Exception as e:
        print(f"Service 1 failed: {e}")

    try:
        print("Service 2: ", qos_system.enforce(service2))  # This should raise an exception
    except Exception as e:
        print(f"Service 2 failed: {e}")
