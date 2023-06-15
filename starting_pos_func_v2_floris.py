import numpy as np
from probability_density_function import pdfunction

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

import matplotlib.pyplot as plt


# resolution_x = (10 * 20 + 1)
# resolution_y = (10 * 20 + 1)
# xx, yy = np.meshgrid(
#     np.linspace(0, 10, resolution_x),
#     np.linspace(0, 10, resolution_y),
#     )
# z = pdfunction(xx, yy, type = 7)


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
    max_radius = max(pdf.shape[0], pdf.shape[1])  # change this line to get the max dimension
    for r in range(1, max_radius):
        points = circle_points(r, center)
        points = [(p[0], p[1]) for p in points if 0 <= p[0] < pdf.shape[0] and 0 <= p[1] < pdf.shape[1]]
        if not points:
            continue
        values = [pdf[p[0], p[1]] for p in points]
        max_value = max(values)
        if max_value <= peak_z / 2:
            mask = create_circular_mask(center, r, pdf.shape)
            mass = np.sum(pdf[mask])
            return r, points[values.index(max_value)], max_value, mass
    raise Exception('No valid points found on any circle')


def speed_list_to_dict(speed_list, squared_speed = True):
    speed_list = np.array(speed_list)

    # Get the unique speeds in the list
    unique_speeds = np.unique(speed_list)

    # Initialize a dictionary to hold the speed lists
    speed_dict = {}

    # Iterate over the unique speeds and create a separate count for each speed
    for speed in unique_speeds:
        speed_dict[speed] = np.sum(speed_list == speed)

    # Sort the dictionary by keys (speeds) in descending order
    sorted_dict = dict(sorted(speed_dict.items(), key=lambda item: item[0], reverse=True))

    if squared_speed:
        power = 2
    else:
        power = 1

    # Calculate the weights for each speed (square of speed)
    weights_dict = {speed: (speed ** power) for speed in sorted_dict.keys()}

    # Calculate the total weight
    total_weight = sum([weights_dict[speed] * count for speed, count in sorted_dict.items()])
    # print(total_weight)

    # Normalize the weights so that they sum up to 1
    normalized_weights_dict = {speed: (weight / total_weight) for speed, weight in weights_dict.items()}
    return sorted_dict, normalized_weights_dict


def distribute_robots_over_peaks(x_mesh, y_mesh, z_mesh, speed_list):
    peaks_ij = get_peaks(z_mesh)

    n_peaks = len(peaks_ij)

    radius_list = []
    masses_peaks = []

    for peak in peaks_ij:
        peak_z = z_mesh[peak[0], peak[1]]
        radius, _, max_val, mass = get_highest_on_circle(z_mesh, peak, peak_z)
        radius_list.append(x_mesh[0, radius])

        masses_peaks.append(mass)

    zipped_lists = list(zip(masses_peaks, radius_list, peaks_ij))
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    masses_peaks, radius_list, peaks_ij = zip(*sorted_lists)

    peaks_xy = [[x_mesh[peak_ij[0], peak_ij[1]], y_mesh[peak_ij[0], peak_ij[1]]] for peak_ij in peaks_ij]

    mass_peaks_array = np.array(masses_peaks)

    mass_fractions_peaks = mass_peaks_array / mass_peaks_array.sum()
    # print(mass_fractions_peaks)
    counts_robot_dict, weights_dict = speed_list_to_dict(speed_list)

    current_mass_robots = np.zeros(n_peaks)
    robots_at_peak = [[] for _ in range(n_peaks)]

    for speed in counts_robot_dict:
        # print(speed)

        if counts_robot_dict[speed] == 0:
            continue

        # flag variable
        continue_next_speed = False

        not_assigned_at_peak = -1

        while counts_robot_dict[speed] > 0:
            for i, mass_frac_peak in enumerate(mass_fractions_peaks):
                if current_mass_robots[i] + weights_dict[speed] <= mass_frac_peak:
                    counts_robot_dict[speed] -= 1
                    current_mass_robots[i] += weights_dict[speed]
                    print(current_mass_robots)
                    robots_at_peak[i].append(speed)
                    not_assigned_at_peak = -1

                    print(f"Assigned a robot with speed {speed} to peak {i}, robots left with this speed: {counts_robot_dict[speed]}")

                elif not_assigned_at_peak == i:  # again at the not assigned peak
                    print(f"Again at peak {i}, continue to next speed")
                    continue_next_speed = True
                    break  # break the for loop

                elif not_assigned_at_peak == -1:
                    print(f"Not assigned at peak: {i}")
                    not_assigned_at_peak = i
                    continue
                else:
                    print("else")

            if continue_next_speed:
                break
    print(robots_at_peak)
    print(counts_robot_dict)

    # add remaining robots
    for speed in counts_robot_dict:
        if counts_robot_dict[speed] == 0:
             continue
        while counts_robot_dict[speed] > 0:
            print(mass_fractions_peaks-current_mass_robots)
            i = np.argmax(mass_fractions_peaks - current_mass_robots)
            counts_robot_dict[speed] -= 1
            current_mass_robots[i] += weights_dict[speed]
            robots_at_peak[i].append(speed)

    # put speed dictionaries for every peak in a list
    speed_dictionaries = []
    for speed_list_output in robots_at_peak:
        speed_dictionaries.append(speed_list_to_dict(speed_list_output, squared_speed = False)[0])

    #get plane bounds
    x_min = np.min(x_mesh)
    x_max = np.max(x_mesh)
    print(x_min, x_max)
    y_min = np.min(y_mesh)
    y_max = np.max(y_mesh)
    print(x_min, x_max)
    # create empty array for robot positions and empty list for robot_speeds
    positions_robots_out = np.empty((0, 2))
    speed_robots_out = []
    x_center_list = []
    y_center_list = []
    # iterate over peaks in x, y coordinates
    for i, peak in enumerate(peaks_xy):

        speed_list_peak = speed_dictionaries[i]
        r_step = np.array(radius_list[i]) * 1.25 / len(speed_list_peak)
        print(f"r_step:{r_step}")
        x_center = peak[0]
        y_center = peak[1]


    # iterate over speed categories
        for j, speed in enumerate(speed_list_peak):
            radius = (j + 1) * r_step
            print(f"radius::{radius}")
            if speed_list_peak[speed] == 0:
                continue
            # iterate over robots in each speed category and generate points on circle
            for k in range(speed_list_peak[speed]):
                stop_after = 0
                while stop_after < 50:  # keep looping until we break
                    random_t = np.random.uniform(0, 2 * np.pi)  # random value between 0 and 2pi
                    x_values = x_center + radius * np.cos(random_t)
                    y_values = y_center + radius * np.sin(random_t)

                    # Check if the generated values fall within the desired range
                    if x_min <= x_values <= x_max and y_min <= y_values <= y_max:
                        print(f"x: {x_values}, y: {y_values} didn't fit, try new random pos.\n")
                        break  # If they do, we break out of the loop
                    else:
                        stop_after += 1
                        continue  # If they don't, we go back to the start of the loop and generate new values

                print(f"x: {x_values}, y: {y_values}")
                positions_robots_out = np.append(positions_robots_out, np.array([[x_values, y_values]]), axis=0)
                speed_robots_out.append(speed)

    return positions_robots_out, speed_robots_out

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


# speed_robots_init = [4,4,4,3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
# print(distribute_robots_over_peaks(xx, yy, z, speed_robots_init))
#
# points = distribute_robots_over_peaks(xx, yy, z, speed_robots_init)[0]
# x_coords, y_coords = zip(*points)

# plt.figure(figsize=(10,10))
# plt.scatter(x_center_list, y_center_list, color='orange')  # Create scatter plot for the centers
# plt.scatter(x_coords, y_coords)  # Create scatter plot
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot of Points')
# plt.xlim([0, 10])
# plt.ylim([0, 10])
#
# plt.show()