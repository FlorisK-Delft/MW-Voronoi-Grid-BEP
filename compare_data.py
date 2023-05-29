import os
import json
import matplotlib.pyplot as plt
import numpy as np

root_dir = 'combined_global_dirs'
map_with_json_files = f'{root_dir}/combined'

json_files = [f for f in os.listdir(map_with_json_files) if f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

plt.figure()

# Create a color map
cmap = plt.get_cmap('viridis')  # change to the colormap you prefer

for json_file in json_files:
    with open(f'{map_with_json_files}/{json_file}', 'r') as f:
        data = json.load(f)

    mw_vor_end_positions_and_v = data["mw_vor_end_positions_and_v"]
    x_values = [item[0] for item in mw_vor_end_positions_and_v]
    y_values = [item[1] for item in mw_vor_end_positions_and_v]
    color_values = [item[2] for item in mw_vor_end_positions_and_v]

    plt.scatter(x_values, y_values, c=color_values, cmap=cmap)  # plot points with color

plt.colorbar(label='Color scale')  # show color scale

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
           alpha=0.5)
plt.colorbar()

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter plot of mw_vor_end_positions_and_v')
plt.show()
