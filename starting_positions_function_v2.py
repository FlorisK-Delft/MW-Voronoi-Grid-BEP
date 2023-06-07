import numpy as np
import matplotlib.pyplot as plt
from probability_density_function import pdfunction, triple_gaussian
from scipy.signal import argrelextrema
from classes import Plane
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from collections import deque


init_plane = Plane([0, 10], [0, 10])
resolution_x = ((init_plane.x_max - init_plane.x_min) * 20 + 1)
resolution_y = ((init_plane.y_max - init_plane.y_min) * 20 + 1)
xx, yy = np.meshgrid(
    np.linspace(*init_plane.return_x_limits(), resolution_x),
    np.linspace(*init_plane.return_y_limits(), resolution_y),
    )
z = pdfunction(xx, yy, type = 6, sigma_x=3.3, sigma_y=3.3)

def get_peaks(z):
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(z, footprint=neighborhood)==z
    local_max = local_max ^ binary_erosion(local_max)
    peaks = np.argwhere(local_max)
    return peaks

def circle_points(radius, center):
    points = []
    for x in range(-radius, radius+1):
        y = int((radius**2 - x**2)**0.5)
        points.extend([(center[0]+x, center[1]+y), (center[0]+x, center[1]-y)])
    return points

def create_circular_mask(center, radius, shape):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
    mask = dist_from_center <= radius
    return mask

def get_highest_on_circle(pdf, center, peak_z):
    for r in range(1, min(center[0], center[1], pdf.shape[0] - center[0], pdf.shape[1] - center[1])):
        points = circle_points(r, center)
        values = [pdf[p[0], p[1]] for p in points]
        max_value = max(values)
        if max_value <= peak_z / 2:
            mask = create_circular_mask(center, r, pdf.shape)
            mass = np.sum(pdf[mask])
            return r, points[values.index(max_value)], max_value, mass
    return None #adjust so it works for ellipses

def return_radius_center(z):
    peaks = get_peaks(z)
    radius_list = []
    masses = []
    for peak in peaks:
        peak_z = z[peak[0], peak[1]]
        radius, _, max_val, mass = get_highest_on_circle(z, peak, peak_z)
        radius_list.append(radius)
        masses.append(mass)
    return radius_list, peaks.tolist(), masses

def initialize_starting_positions_v2(x_mesh, y_mesh, z_mesh, speed_list):
    positions_robots = np.empty((0, 2))
    speed_robots = []

    num_speeds = len(np.unique(speed_list))
    speed_counts = {}
    for speed in speed_list:
        if speed in speed_counts:
            # If the speed is already in the dictionary, increment its count
            speed_counts[speed] += 1
        else:
            # If the speed is not in the dictionary, add it with a count of 1
            speed_counts[speed] = 1

    peaks = get_peaks(z_mesh)
    radius_list = []
    masses = []

    for peak in peaks:
        peak_z = z_mesh[peak[0], peak[1]]
        radius, _, max_val, mass = get_highest_on_circle(z, peak, peak_z)
        radius_list.append(radius)
        masses.append(mass)

    zipped_lists = list(zip(masses, radius_list, peaks))
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    masses_sorted, radius_list_sorted, peaks_sorted = zip(*sorted_lists)
    mass_temp = np.array(masses_sorted)

    mass_fractions = mass_temp / mass_temp.sum()
    weight_list = [robot / np.array(speed_list).sum() for robot in speed_list]

    robots = sorted(zip(speed_list, weight_list), key=lambda x: x[0], reverse=True)

    # Combine speeds and weights into a list of tuples and sort by speed
    robots = sorted(zip(speed_list, weight_list), key=lambda x: x[0], reverse=True)

    # Initialize assignments and peak load
    assignments = [[] for _ in mass_fractions]
    combined_weight_robots = np.zeros(len(mass_fractions))

    # Iterate through each speed category
    robots = sorted(zip(speed_list, weight_list), key=lambda x: x[0], reverse=True)

    # Initialize assignments and peak load
    assignments = [[] for _ in mass_fractions]
    combined_weight_robots = np.zeros(len(mass_fractions))

    robots_not_assigned = []


    # Iterate through each speed category [4,3,2,1]
    for speed in sorted(set(speed_list), reverse=True):
        print(f"starting with speed category {speed}")
        # Filter robots for the current speed category
        speed_robots = [robot for robot in robots if robot[0] == speed]
        # Distribute robots of this speed category to the peaks
        # unassigned = True
        while speed_robots:  # loop until no robots of this speed left [1, 0.23
            for i, mass_fraction in enumerate(mass_fractions):
                assigned = False
                if not speed_robots:  # if no robots of this speed left, break the loop
                    break

                robot = speed_robots[0]  # take the first robot of this speed
                # If adding this robot would not exceed the peak's mass fraction
                if combined_weight_robots[i] + robot[1] <= mass_fraction:
                    assignments[i].append(robot[0])
                    combined_weight_robots[i] += robot[1]
                    speed_robots.remove(robot)  # remove the assigned robot from the list
                    assigned = True  # a robot was assigned

                if not assigned:
                    print(f"{len(speed_robots)} robot('s) with speed: {speed} are not assigned") # no robot was assigned in the loop
                    robots_not_assigned.append(robot)
                    speed_robots.remove(robot)
                    continue  # break the while loop

        # Remove distributed robots from the original list
        robots = [robot for robot in robots if robot[0] != speed]



    for speed in sorted(set(speed_list), reverse=True):
        speed_robots = [robots_not_assigned for robots_not_assigned in robots_not_assigned if robots_not_assigned[0] == speed]

        while speed_robots:
            robot = speed_robots[0]
            # Find the index of the peak with the highest remaining mass fraction
            i = np.argmax(mass_fractions - combined_weight_robots)
            assignments[i].append(robot[0])
            combined_weight_robots[i] += robot[1]
            speed_robots.remove(robot)

        print(assignments)

    return assignments

speed_robots_init = [4,4,4,3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]

initialize_starting_positions_v2(xx, yy, z, speed_robots_init)

