import numpy as np
from probability_density_function import pdfunction

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion



resolution_x = (10 * 20 + 1)
resolution_y = (10 * 20 + 1)
x, y = np.meshgrid(
    np.linspace(0, 10, resolution_x),
    np.linspace(0, 10, resolution_y),
    )
z = pdfunction(x, y, type = 6, sigma_x=3.3, sigma_y=3.3)


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

def speed_list_to_dict(speed_list, squared_speed = False):
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
    total_weight = sum([weight * count for speed, (weight, count) in zip(weights_dict.keys(), sorted_dict.items())])

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
        radius, _, max_val, mass = get_highest_on_circle(z, peak, peak_z)
        radius_list.append(x_mesh[0, radius])

        masses_peaks.append(mass)

    zipped_lists = list(zip(masses_peaks, radius_list, peaks_ij))
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    masses_peaks, radius_list, peaks_ij = zip(*sorted_lists)

    peaks_xy = [[x_mesh[peak_ij[0], peak_ij[1]], y_mesh[peak_ij[0], peak_ij[1]]] for peak_ij in peaks_ij]

    mass_peaks_array = np.array(masses_peaks)

    mass_fractions_peaks = mass_peaks_array / mass_peaks_array.sum()

    counts_robot_dict, weights_dict = speed_list_to_dict(speed_list)

    current_mass_robots = np.zeros(n_peaks)
    robots_at_peak = [[] for _ in range(n_peaks)]

    for speed in counts_robot_dict:
        print(speed)

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


speed_robots_init = [4,4,4,3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
distribute_robots_over_peaks(x, y, z, speed_robots_init)

print("iij")