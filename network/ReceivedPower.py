# -*- coding: utf-8 -*-
"""
Received Power Calculator based on the Okamura-Hata model, including scenario support for UAVs.

@author: Jose Matamoros (Updated by major)
"""
from math import log


def receivedPower(pt, d, hb, scenario="urban", f=700):
    '''
    Calculates the received power based on the Okamura-Hata model, with support for different scenarios.

    Parameters
    ----------
    pt : float
        Transmitted Power [dB]
    d : float
        Distance [m]
    hb : float
        Base Station or UAV altitude [m]
    scenario : str, optional
        Scenario for environment (urban, suburban, rural, etc.). Default is 'urban'.
    f : float, optional
        Carrier Frequency in MHz (default is 700 MHz)

    Returns
    -------
    float
        Received Power [dB]
    '''

    hm = 2  # Mobile terminal height (user height)

    # Adjust for the scenario by modifying environment-dependent values
    scenario_params = {
        'urban': {'am': 0.8, 'b': 5.83, 'c': 16.33},
        'suburban': {'am': 0.5, 'b': 4.0, 'c': 12.0},
        'rural': {'am': 0.3, 'b': 3.0, 'c': 10.0},
    }

    if scenario not in scenario_params:
        raise ValueError("Unsupported scenario. Supported values: 'urban', 'suburban', 'rural'.")

    am = (1.1 * log(f, 10) - 0.7) * hm - (1.56 * log(f, 10) - 0.8)
    params = scenario_params[scenario]
    a = 69.55 + 26.16 * log(f, 10) - 13.82 * log(hb, 10) - am
    b = params['b']
    c = params['c']

    # Path loss calculation
    path_loss = a + b * log(d, 10) + c

    # Calculate received power: Received Power = Transmitted Power - Path Loss
    return pt - path_loss


if __name__ == "__main__":
    # Test cases for different scenarios
    print(f"Urban Scenario: {receivedPower(0, 0.04, 2, 'urban')}")
    print(f"Suburban Scenario: {receivedPower(10, 100, 10, 'suburban')}")
    print(f"Rural Scenario: {receivedPower(20, 500, 30, 'rural')}")
    print(f"Custom Frequency (900 MHz) in Urban: {receivedPower(20, 500, 10, 'urban', f=900)}")
