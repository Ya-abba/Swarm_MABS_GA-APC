# -*- coding: utf-8 -*-
"""
Enhanced Mobility Models with Boundary Checks and Data Rate Considerations

@author: major
"""

import random
import numpy as np

class Location:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Location({self.x}, {self.y}, {self.z})"

    def distance_to(self, other_location):
        return np.sqrt((self.x - other_location.x) ** 2 + 
                       (self.y - other_location.y) ** 2 + 
                       (self.z - other_location.z) ** 2)


class User:
    def __init__(self, initial_location, data_rate_requirement=None):
        '''
        Initializes a user with an optional data rate requirement.
        '''
        self.location = initial_location
        self.path = []
        self.data_rate_requirement = data_rate_requirement  # User's data rate requirement (e.g., 1 Mbps, 2 Mbps, or 5 Mbps)

    def set_path(self, path):
        self.path = path

    def move(self):
        if self.path:
            self.location = self.path.pop(0)
        else:
            print("No more locations in path. Staying at current location.")


def random_waypoint_mobility(user, area_size, speed, time_step, pause_probability=0.1):
    '''
    Update user location using the Random Waypoint Mobility Model with pause capability.

    Parameters
    ----------
    pause_probability : float
        Probability that the user will pause instead of moving.

    Returns
    -------
    Updated location of the user.
    '''
    if random.random() < pause_probability:
        return user.location  # User pauses at the current location

    dest_x = random.uniform(0, area_size[0])
    dest_y = random.uniform(0, area_size[1])
    dest_z = random.uniform(0, area_size[2])

    destination = Location(dest_x, dest_y, dest_z)
    distance = user.location.distance_to(destination)

    if distance > 0:
        move_x = (dest_x - user.location.x) / distance * speed * time_step
        move_y = (dest_y - user.location.y) / distance * speed * time_step
        move_z = (dest_z - user.location.z) / distance * speed * time_step
    else:
        move_x, move_y, move_z = 0, 0, 0

    # Update the user's location with boundary checks
    user.location.x = np.clip(user.location.x + move_x, 0, area_size[0])
    user.location.y = np.clip(user.location.y + move_y, 0, area_size[1])
    user.location.z = np.clip(user.location.z + move_z, 0, area_size[2])

    return user.location


def reference_point_group_mobility(group, reference_point, speed, area_size, time_step, group_deviation=0.5):
    '''
    Update user group locations using the Reference Point Group Mobility Model with added randomness.

    Parameters
    ----------
    group_deviation : float
        Max deviation in movement direction for users relative to the reference point.

    Returns
    -------
    Updated locations of the group of users.
    '''
    # Move the reference point using Random Waypoint Mobility Model
    ref_x = random.uniform(0, area_size[0])
    ref_y = random.uniform(0, area_size[1])
    ref_z = random.uniform(0, area_size[2])

    destination = Location(ref_x, ref_y, ref_z)
    distance = reference_point.distance_to(destination)

    if distance > 0:
        move_x = (ref_x - reference_point.x) / distance * speed * time_step
        move_y = (ref_y - reference_point.y) / distance * speed * time_step
        move_z = (ref_z - reference_point.z) / distance * speed * time_step
    else:
        move_x, move_y, move_z = 0, 0, 0

    # Update the reference point's location with boundary checks
    reference_point.x = np.clip(reference_point.x + move_x, 0, area_size[0])
    reference_point.y = np.clip(reference_point.y + move_y, 0, area_size[1])
    reference_point.z = np.clip(reference_point.z + move_z, 0, area_size[2])

    # Move each user relative to the reference point with added deviation
    for user in group:
        deviation_x = random.uniform(-group_deviation, group_deviation)
        deviation_y = random.uniform(-group_deviation, group_deviation)
        deviation_z = random.uniform(-group_deviation, group_deviation)

        user.location.x = np.clip(user.location.x + move_x + deviation_x, 0, area_size[0])
        user.location.y = np.clip(user.location.y + move_y + deviation_y, 0, area_size[1])
        user.location.z = np.clip(user.location.z + move_z + deviation_z, 0, area_size[2])

    return [user.location for user in group]


# Example usage:
if __name__ == "__main__":
    # Random Waypoint Mobility Model example
    user1 = User(Location(0, 0, 0), data_rate_requirement=5e6)  # User with a 5 Mbps data rate requirement
    area_size = (100, 100, 50)
    speed = 10
    time_step = 1

    print("RWPMM:")
    for _ in range(5):
        new_location = random_waypoint_mobility(user1, area_size, speed, time_step)
        print(f"Updated location: {new_location}")

    # Reference Point Group Mobility Model example
    user_group = [
        User(Location(10, 10, 5), data_rate_requirement=2e6),
        User(Location(12, 15, 5), data_rate_requirement=1e6),
        User(Location(8, 12, 5), data_rate_requirement=5e6),
    ]
    ref_point = Location(10, 10, 5)

    print("\nRPGM:")
    for _ in range(5):
        updated_locations = reference_point_group_mobility(user_group, ref_point, speed, area_size, time_step)
        print(f"Updated group locations: {updated_locations}")
