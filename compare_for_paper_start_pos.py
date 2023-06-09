import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
import textwrap

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
    number_of_runs = 0

    json_files = [f for f in os.listdir(f"{dir}/combined") if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

    for json_file in json_files:
        number_of_runs += 1
        with open(f'{dir}/combined/{json_file}', 'r') as f:
            data = json.load(f)

        list_mw_vor_end_time.append(data["mw_vor_end_time"])
        list_number_iterations.append(len(data["mw_vor_response_time_list"]))
        list_mw_vor_start_time.append(data["mw_vor_start_time"])

    return list_mw_vor_start_time, list_mw_vor_end_time, list_number_iterations, number_of_runs


list_mw_vor_start_time_random, list_mw_vor_end_time_random, list_number_iterations_random, number_of_runs_random = return_list_of_data(
    dir_random_start)
list_mw_vor_start_time_chosen, list_mw_vor_end_time_chosen, list_number_iterations_chosen, number_of_runs_chosen = return_list_of_data(
    dir_chosen_start)

print(list_mw_vor_end_time_chosen)
print(list_mw_vor_end_time_random)

print(
    f"Average number of iterations for random: {sum(list_number_iterations_random) / len(list_number_iterations_random)}")
print(
    f"Average number of iterations for chosen: {sum(list_number_iterations_chosen) / len(list_number_iterations_chosen)}")

# ------------------------------------------------------------------------
# Start of making the boxplot


font_path_regular = './lmroman7-regular.otf'
font_prop_regular = fm.FontProperties(fname=font_path_regular)

font_path_bold = './lmroman7-bold.otf'
font_prop_bold = fm.FontProperties(fname=font_path_bold)

fig = plt.figure(figsize=(10 / 1.1, 6 / 1.1))
ax = fig.add_subplot(111)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(3))
ax.yaxis.grid(True, which='both', linestyle='-', color='lightgrey', alpha=0.7)

box_width = 0.4

# Plot data
boxplot_data = [list_mw_vor_end_time_random, list_mw_vor_end_time_chosen]
bp = ax.boxplot(boxplot_data, widths=box_width, medianprops={'linewidth': 3})

labels = [f'Random start positions, $n={number_of_runs_random}$',
          f'Chosen start positions, $n={number_of_runs_chosen}$']
ax.set_xticklabels(labels, fontproperties=font_prop_regular, fontsize=16)

ax.set_ylabel('Average response time\u00B2', fontproperties=font_prop_regular, fontsize=16)

# title = 'Comparison of average response time²\nof final distribution'  # nog mee nemen hier tT met een streepje
# wrapped_title = textwrap.fill(title, 37)  # Adjust the line width as needed
# ax.set_title(wrapped_title, fontproperties=font_prop_bold, fontsize=20)

plt.savefig(f"box_plot_random_and_chosen.png", dpi=250)  # dpi is the resolution of each png
plt.show()
