import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

root_dir = 'combined_global_dirs'
map_with_json_files = f'{root_dir}/combined'

json_files = [f for f in os.listdir(map_with_json_files) if f.endswith('.json') and f != 'global_mesh_data.json']

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

    mw_vor_end_positions_and_v = data["start_positions_and_v"]
    x_values = [item[0] for item in mw_vor_end_positions_and_v]
    y_values = [item[1] for item in mw_vor_end_positions_and_v]
    speed_values = [item[2] for item in mw_vor_end_positions_and_v]

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

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter plot of start_positions_and_v')

plt.subplots_adjust(bottom=0.2) # provide space at the bottom of the plot for the legend

plt.savefig("color_start_pos.png", dpi=250)  # dpi is the resolution of each png
plt.show()
