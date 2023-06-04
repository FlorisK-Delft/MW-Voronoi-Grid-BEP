import numpy as np
import matplotlib.pyplot as plt
from probability_density_function import pdfunction, triple_gaussian
from scipy.signal import argrelextrema
from classes import Plane
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion



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

peaks = get_peaks(z)
import numpy as np
from scipy.ndimage import gaussian_filter

# Create an example PDF
pdf = np.random.rand(201, 201)

# Apply a smoothing filter to make the values more gradual
pdf = gaussian_filter(pdf, sigma=3)

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

for peak in peaks:
    peak_z = z[peak[0], peak[1]]
    print(get_highest_on_circle(z, peak, peak_z))

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

# def return_radii():
#     radius_list = return_radius_center(z)[0]
#     larger_radius_list = np.array(radius_list)*1.5
#     surface_areas = [np.pi * larger_radius**2 for larger_radius  in larger_radius_list]
#     surface_area_robot = np.array(surface_areas)/len(velo)
#
# return_radii()

print(return_radius_center(z))

def initialize_starting_positions(x_mesh, y_mesh, z_mesh, speed_list=[3,3,3,3,3,2,2,2,1,1]):

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


    for i, peak in enumerate(peaks):
        mass_fraction = mass_temp[i]/mass_temp.sum()
        mass_temp[i] = 0 # isn't counted for the weight for assigning the next robot
        rounded_num_robots_current_peak = int(round(mass_fraction*len(speed_list)))

        speed_counts_this_peak = {}
        for i, speed in enumerate(speed_counts):
            count_this_speed = speed_counts[speed]
            this_speed_this_peek = int(round(count_this_speed*mass_fraction))
            speed_counts_this_peak[speed] = this_speed_this_peek

            # save how much robots are left:
            speed_counts[speed] = count_this_speed - this_speed_this_peek

        # print(speed_counts_this_peak)
        r_step = x_mesh[0,radius_list_sorted[i]] * 1.25 / len(speed_counts)
        x_center = x_mesh[peak[0], peak[1]]
        y_center = y_mesh[peak[0], peak[1]]

        speed_counts_this_peak = {k: speed_counts_this_peak[k] for k in sorted(speed_counts_this_peak)}

        for i, speed in enumerate(speed_counts_this_peak):
            (i+1)*r_step
            t_values = np.linspace(0, 2 * np.pi, 100)

            x_values = x_center + radius * np.cos(t_values)
            y_values = y_center + radius * np.sin(t_values)




    print(masses_sorted, radius_list_sorted, peaks_sorted)

initialize_starting_positions(xx, yy, z)


# peak = [50, 100]
# peak_z = pdf[peak[0], peak[1]]
# print(get_highest_on_circle(pdf, peak, peak_z))
#
# #find peaks
# neighborhood = generate_binary_structure(2,2)
# local_max = maximum_filter(z, footprint=neighborhood)==z
# local_max = local_max ^ binary_erosion(local_max)
# peaks = np.argwhere(local_max)
# print(peaks)
# #compute fwhm for all the peaks
# for peak in peaks:
#     half_max = z[peak[0]][peak[1]]/2
#     directions = ["x", "y"]
#     for direction in directions:
#         if direction == "x":
#             cross_section = z[peak[0], :]
#         elif direction == "y":
#             cross_section = z[:, peak[1]]
#
#         left_or_above_peak = cross_section[:peak[1]]
#         try:
#             fwhm_start = max(idx for idx, val in enumerate(left_or_above_peak) if val > half_max)
#         except ValueError:  # Raised if no values are above half_max
#             fwhm_start = 0
#         right_or_under_peak = cross_section[peak[1]:]
#         try:
#             fwhm_end = peak[1] + min(idx for idx, val in enumerate(right_or_under_peak) if val < half_max)
#         except ValueError:  # Raised if no values are below half_max
#             fwhm_end = len(cross_section)
#         fwhm = fwhm_end - fwhm_start
#         if direction == "x":
#             print(f"fwhm_x: {fwhm}")
#         elif direction == "y":
#             print(f"fwhm_y: {fwhm}")
#
# weights = []
# FWHM = 21
#
# for peak in peaks:
#     x, y = peak
#     # Define the region of interest around the peak using the FWHM
#     roi = z[max(0, x - FWHM // 2):min(z.shape[0], x + FWHM // 2),
#           max(0, y - FWHM // 2):min(z.shape[1], y + FWHM // 2)]
#
#     # Calculate the weight of the peak by summing the values in the region of interest
#     weight = np.sum(roi)
#     weights.append(weight)
# print(np.sum(weights))
# print(np.sum(z))
# #fwhm_list = []
# #for peak_coords in peaks:
#    # x_center, y_center = peak_coords
#    # half_max =
#   #  fwhm_list.append(fwhm)
#
#
# #print(fwhm_list)
# #def coordinates(z):
#  #   peaks_indices = signal.argrelextrema(z, np.greater)[0]
#   #  peak_values = z[peaks_indices]
#    # return peak_values, peaks_indices
#
#
#
# #x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, (x_max-x_min)+1), np.linspace(y_min, y_max, (y_max-y_min)+1))
#     #positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
#     #velocities = np.array([])
#     #fig = plt.figure()
#     #ax = fig.add_subplot(111)
#     #ax.plot(x_mesh, y_mesh, ls="None", marker=".")
#     #plt.show()