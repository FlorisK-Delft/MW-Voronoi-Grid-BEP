import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import numpy as np
from starting_positions_function import return_radius_center

show_end_pos = True
show_density_function = True
show_circles = True

root_dir = 'combined_global_dirs_run30mei_3punten_zelfde_groote'
map_with_json_files = f'{root_dir}/combined'

json_files = [f for f in os.listdir(map_with_json_files) if f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

# Initialize min and max with infinity and negative infinity so any number from the data will replace them
min_time = float('inf')
max_time = float('-inf')

# Find the min and max values
for json_file in json_files:
    with open(f'{map_with_json_files}/{json_file}', 'r') as f:
        data = json.load(f)
        time = data["mw_vor_end_time"]
        min_time = min(min_time, time)
        max_time = max(max_time, time)

plt.figure()

# Define colors for different speeds
colors = {1.0: 'red', 2.0: 'black', 3.0: 'blue'}

for json_file in json_files:
    with open(f'{map_with_json_files}/{json_file}', 'r') as f:
        data = json.load(f)

    if show_end_pos:
        pos = data["mw_vor_end_positions_and_v"]
    else:
        pos = data["start_positions_and_v"]

    x_values = [item[0] for item in pos]
    y_values = [item[1] for item in pos]
    speed_values = [item[2] for item in pos]

    # Normalize time to get an alpha value in the range [0.2, 0.8]
    alpha = 1 * (data["mw_vor_end_time"] - min_time) / (max_time - min_time) #+ 0.2

    for x, y, v in zip(x_values, y_values, speed_values):
        plt.scatter(x, y, color=colors[v], alpha=alpha)  # plot points with color and transparency

# Create legend
red_patch = mpatches.Patch(color='red', label='Speed 1')
green_patch = mpatches.Patch(color='black', label='Speed 2')
blue_patch = mpatches.Patch(color='blue', label='Speed 3')
plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.legend(handles=[red_patch, green_patch, blue_patch], bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)

# plt.xlabel('X values')
# plt.ylabel('Y values')
if show_end_pos:
    plt.title('End positions shorted on avg reaction speed')
else:
    plt.title('Start positions shorted on avg reaction speed')

plt.subplots_adjust(bottom=0.2) # provide space at the bottom of the plot for the legend

if show_density_function:
    # Load the JSON data
    with open(f'{root_dir}/global_mesh_data.json', 'r') as f:
        data = json.load(f)

    # Get the z_mesh data
    z_mesh_list = data['z_mesh']

    # Convert the list to a numpy array
    z_mesh = np.array(z_mesh_list)

    # plot the gradient of the pdf (this is z)
    plt.imshow(z_mesh, origin='lower',
               extent=(0, 10, 0, 10),
               alpha=0.25)
    plt.colorbar()

if show_circles:
    # def plot_circles(radius, peak_indices):
    #     fig, ax = plt.subplots()
    #
    #     # Plot the mesh grid
    #     # Replace 'mesh' with your own mesh grid data
    #     # For example, if you have a 2D array of values, use ax.imshow(mesh)
    #     ax.imshow(mesh, cmap='gray')
    #
    #     # Plot circles around the peaks
    #     for idx in peak_indices:
    #         peak_x, peak_y = idx
    #         circle = Circle((peak_x, peak_y), radius, color='red', fill=False)
    #         ax.add_patch(circle)

    with open(f'{root_dir}/global_mesh_data.json', 'r') as f:
        data = json.load(f)

    # Get the z_mesh data
    z_mesh_list = data['z_mesh']
    x_mesh_list = data['x_mesh']
    y_mesh_list = data['y_mesh']

    # Convert the list to a numpy array
    z_mesh = np.array(z_mesh_list)
    x_mesh = np.array(x_mesh_list)
    y_mesh = np.array(y_mesh_list)


    radius_list, peaks, masses = return_radius_center(z_mesh)
    for i, peak in enumerate(peaks):
        print(peak)
        print(radius_list[i])
        peak_x = x_mesh[peak[0], peak[1]]
        peak_y = y_mesh[peak[0], peak[1]]
        circle = Circle((peak_x, peak_y), x_mesh[0, radius_list[i]], color='orange', fill=False, linewidth=masses[i]*10)
        plt.gca().add_patch(circle)


if show_end_pos:
    plt.savefig(f"{root_dir}/color_end_pos.png", dpi=250)  # dpi is the resolution of each png
else:
    plt.savefig(f"{root_dir}/color_start_pos.png", dpi=250)  # dpi is the resolution of each png

plt.show()
