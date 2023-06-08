import os
import json
import matplotlib.pyplot as plt
import numpy as np

dir_random_start = "combined_global_dirs_run4jun_andere_sigam_random"
dir_chosen_start = "combined_global_dirs_chosen_starts_3peaks_diff_sigma"


def get_z_mesh_from(dir):
    if not os.path.isdir(f"{dir}/combined"):
        print(f"File {dir}/combined does not exist. Please run 'sort_runs.py' first.")
        exit(1)

    with open(f'{dir}/global_mesh_data.json', 'r') as f:
        data = json.load(f)

    # Get the z_mesh data
    z_mesh_list = data['z_mesh']

    # Convert the list to a numpy array
    z_mesh = np.array(z_mesh_list)

    return z_mesh

def plot_meshes(mesh1, mesh2, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(mesh1, cmap='hot', interpolation='nearest', origin='lower')
    axs[0].set_title(title1)

    axs[1].imshow(mesh2, cmap='hot', interpolation='nearest', origin='lower')
    axs[1].set_title(title2)

    plt.show()

def stop_criterion_dir(dir):
    json_files = [f for f in os.listdir(f"{dir}/combined") if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']
    list_stop_criterion = []
    for json_file in json_files:
        with open(f'{dir}/combined/{json_file}', 'r') as f:
            data = json.load(f)

        list_stop_criterion.append(data["stop_criterion"])

    # check if data is "clean", so all values are the same, otherwise show the different values.
    unique_values = set(list_stop_criterion)
    if len(unique_values) == 1:
        return list(unique_values)[0]
    else:
        print(f"The list contains these unique values: {', '.join(map(str, unique_values))}")
        exit(1)


def relative_difference(a, b):
    return np.abs((a - b) / ((a + b) / 2))

# Load meshes
z_mesh_random_start = get_z_mesh_from(dir_random_start)
z_mesh_chosen_start = get_z_mesh_from(dir_chosen_start)

# If the maximum relative difference is more than 1%, plot the meshes --> 1% due to small rounding errors etc.
if np.max(relative_difference(z_mesh_random_start, z_mesh_chosen_start)) > 0.01:
    plot_meshes(z_mesh_random_start, z_mesh_chosen_start, dir_random_start, dir_chosen_start)
    print("You can't compare if the maximum relative difference between density functions is more than 1%.")
    exit(1)
elif stop_criterion_dir(dir_random_start) != stop_criterion_dir(dir_chosen_start):
    print("Stop criterion is not the same for both directories.")
    exit(1)
else:
    print("Meshes overlap for >99%, we are able to compare them.\nContinue script.\n")

def return_list_of_data(dir):
    list_mw_vor_start_time = []
    list_mw_vor_end_time = []
    list_number_iterations = []

    json_files = [f for f in os.listdir(f"{dir}/combined") if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

    for json_file in json_files:
        with open(f'{dir}/combined/{json_file}', 'r') as f:
            data = json.load(f)

        list_mw_vor_end_time.append(data["mw_vor_end_time"])
        list_number_iterations.append(len(data["mw_vor_response_time_list"]))
        list_mw_vor_start_time.append(data["mw_vor_start_time"])

    return list_mw_vor_start_time, list_mw_vor_end_time, list_number_iterations

list_mw_vor_start_time_random, list_mw_vor_end_time_random, list_number_iterations_random = return_list_of_data(dir_random_start)
list_mw_vor_start_time_chosen, list_mw_vor_end_time_chosen, list_number_iterations_chosen = return_list_of_data(dir_chosen_start)

print(list_mw_vor_end_time_chosen)
print(list_mw_vor_end_time_random)

print(f"Numer of iterations for random: {sum(list_number_iterations_random) / len(list_number_iterations_random)}")
print(f"Numer of iterations for chosen: {sum(list_number_iterations_chosen) / len(list_number_iterations_chosen)}")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

boxplot_data = [list_mw_vor_end_time_random, list_mw_vor_end_time_chosen]
ax.boxplot(boxplot_data, patch_artist=False)

ax.set_xticklabels(['Random Start Time', 'Chosen Start Time'])

ax.set_ylabel('End Time')

ax.set_title('Comparison of End Times')

plt.show()
