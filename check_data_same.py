import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt

from probability_density_function import pdfunction


def generate_dir_name(chosen_or_random, fit_type):
    # First, we need to get the correct prefix based on the fit type.
    prefix_dict = {
        1: '2eclipse_',
        2: '1single_',
        3: '2equal_',
        4: '3equal_',
        5: '2unequal_height_',
        6: '3unequal_height_',
        7: '3unequal_sigma_',
        8: '2unequal_sigma_',
        9: '4equal_',
        10: '4unequal_height_',
        11: '4unequal_sigma_'
    }

    prefix = prefix_dict.get(fit_type, '')
    dir_name = f'{prefix}{chosen_or_random}_type{fit_type}'

    return dir_name

def plot_start_positions(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        pos = data["start_positions_and_v"]

        x_values = [item[0] for item in pos]
        y_values = [item[1] for item in pos]
        speed_values = [item[2] for item in pos]

        # Plot points with colors mapped from speed values
        plt.scatter(x_values, y_values, c=speed_values, cmap='viridis')

    except KeyError:
        print(f"'start_positions_and_v' not found in {json_file}. Skipping this file.")
        pass


def get_mesh_from(dir):
    with open(f'{dir}/global_mesh_data.json', 'r') as f:
        data = json.load(f)

    # Get the mesh data
    x_mesh = np.array(data['x_mesh'])
    y_mesh = np.array(data['y_mesh'])
    z_mesh = np.array(data['z_mesh'])

    return x_mesh, y_mesh, z_mesh

def relative_difference(a, b):
    return np.abs((a - b) / ((a + b) / 2))

def get_stop_criterion_from(dir):
    with open(f'{dir}/data_run_0.json', 'r') as f:
        data = json.load(f)

    stop_criterion = data.get('stop_criterion')

    return stop_criterion

def calculate_best_fit(x_mesh, y_mesh, z_mesh):
    best_fit_type = None
    best_fit_diff = float('inf')

    for type in range(1, 12):  # types 1 to 11 inclusive
        z_mesh_test = pdfunction(x_mesh, y_mesh, type)
        diff = np.max(relative_difference(z_mesh, z_mesh_test))

        if diff < best_fit_diff:
            best_fit_diff = diff
            best_fit_type = type

        # If we have a match that is good enough, break the loop
        if best_fit_diff < 0.001:
            break

    if best_fit_diff >= 0.001:
        best_fit_type = None  # return None if no match found with at least 99.9% accuracy

    return best_fit_type

def check_data_path(dir=None):
    plot_z_seperate = False

    if dir is None:
        dir = input("Give the path of the folder that needs to be checked:")

    # If there exists a folder named 'combined', delete it.
    combined_dir = os.path.join(dir, 'combined')
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)

    subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    # Let's check that there are subdirectories to process.
    if not subdirs:
        print("Not multiple subdirectories found, subdirs = dir now")
        subdirs = [dir]



    # Let's assume that the global_mesh_data.json in the first directory is our base file.
    base_file_path = os.path.join(subdirs[0], 'global_mesh_data.json')

    # Let's check that this file exists.
    if not os.path.exists(base_file_path):
        print("No global_mesh_data.json file in the first directory.")
        exit()

    # Get base mesh data
    base_x_mesh, base_y_mesh, base_z_mesh = get_mesh_from(subdirs[0])

    # Now let's compare this file to the same file in all other directories.
    for subdir in subdirs[1:]:
        file_to_compare = os.path.join(subdir, 'global_mesh_data.json')
        if not os.path.exists(file_to_compare):
            print(f"No global_mesh_data.json file in directory {subdir}.")
            exit()
        else:
            compare_x_mesh, compare_y_mesh, compare_z_mesh = get_mesh_from(subdir)
            if np.max(relative_difference(base_z_mesh, compare_z_mesh)) > 0.001:
                raise ValueError(f"Data in 'z_mesh' in global_mesh_data.json in {subdirs[0]} and {subdir} are not at least 99% similar.")
            else:
                print("Z mesh is for at least 99.9% the same.")

    # Check 'data_run_0.json' files for 'stop_criterion'
    base_file_path = os.path.join(subdirs[0], 'data_run_0.json')

    # Let's check that this file exists.
    if not os.path.exists(base_file_path):
        print("No data_run_0.json file in the first directory.")
        exit()

    # Get base stop_criterion
    base_stop_criterion = get_stop_criterion_from(subdirs[0])

    if base_stop_criterion is None:
        raise ValueError(f"No 'stop_criterion' in data_run_0.json in {subdirs[0]}.")

    # Now let's compare 'stop_criterion' in this file to the same file in all other directories.
    for subdir in subdirs[1:]:
        file_to_compare = os.path.join(subdir, 'data_run_0.json')
        if not os.path.exists(file_to_compare):
            print(f"No data_run_0.json file in directory {subdir}.")
            exit()
        else:
            compare_stop_criterion = get_stop_criterion_from(subdir)
            if compare_stop_criterion is None:
                raise ValueError(f"No 'stop_criterion' in data_run_0.json in {subdir}.")
            elif base_stop_criterion != compare_stop_criterion:
                raise ValueError(f"'stop_criterion' in data_run_0.json in {subdirs[0]} and {subdir} are not identical.")

    print(f"All 'data_run_0.json' files have the same 'stop_criterion': {base_stop_criterion}")

    if plot_z_seperate == True:
        # Plotting
        plt.figure()
        plt.imshow(base_z_mesh, origin='lower', extent=(np.min(base_x_mesh), np.max(base_x_mesh), np.min(base_y_mesh), np.max(base_y_mesh)), alpha=0.5)
        plt.colorbar()
        plt.show()

    if dir.startswith("data"):
        shutil.copy2(os.path.join(subdirs[0], 'global_mesh_data.json'), os.path.join(dir, 'global_mesh_data.json'))




    # Before plotting
    plt.figure()

    # Plot base mesh
    plt.imshow(base_z_mesh, origin='lower',
               extent=(np.min(base_x_mesh), np.max(base_x_mesh), np.min(base_y_mesh), np.max(base_y_mesh)), alpha=0.5)

    # Plot start positions for each json file in each subdirectory
    for subdir in subdirs:
        json_files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and f.endswith('.json')]
        for json_file in json_files:
            plot_start_positions(os.path.join(subdir, json_file))

    plt.colorbar(ticks=range(5), label='Speed')
    plt.show()



    # call this function at the end of your script
    best_fit = calculate_best_fit(base_x_mesh, base_y_mesh, base_z_mesh)
    print(f"The best fit is with pdf type: {best_fit}")




    dir_to_move_to = None
    if dir.startswith("data"):
        print("You are already in the data dir.")
    else:
        chosen_or_random = input("R for random, C for chosen, N for None / fault:")

        if chosen_or_random == "R":
            print("Chosen option: Random.")
            if best_fit is not None:
                dir_to_move_to = generate_dir_name("random", best_fit)
            else:
                print("No suitable fit found, won't copy data over")
        elif chosen_or_random == "C":
            print("Chosen option: Chosen.")
            if best_fit is not None:
                dir_to_move_to = generate_dir_name("chosen", best_fit)
            else:
                print("No suitable fit found, won't move data over")
        else:
            print("Chosen option: None / Fault.")

    if dir_to_move_to is not None:
        for subdir in subdirs:
            subdir_name = os.path.basename(subdir)
            new_dir_name = dir_to_move_to
            new_dir_path = os.path.join('data', new_dir_name, subdir_name)

            # Create new directory if it doesn't exist
            os.makedirs(os.path.join('data', new_dir_name), exist_ok=True)

            # Move subdir to the new directory
            shutil.move(subdir, new_dir_path)

# check_data_path()