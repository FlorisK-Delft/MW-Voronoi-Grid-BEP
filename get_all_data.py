import os
import json
import shutil
import glob

from check_data_same import check_data_path
from sort_runs import sort_runs
from compare_for_paper_start_pos import compare_random_chosen
from color_scale_start_pos import plot_end_pos, plot_z_mesh
from functions_for_lloyds import get_lloyds_data_from_type, generating_loyds_table

base_dir = 'data'

def generate_dir_name(chosen_or_random, fit_type):
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


def list_dir_names(type_pdf, chosen=True, random=True):
    if isinstance(type_pdf, int):
        type_pdf = [type_pdf]
    elif not isinstance(type_pdf, list):
        raise ValueError("type_pdf must be an integer or a list of integers.")

    list_dir_names = []

    evaluate_c_r = []
    if chosen:
        evaluate_c_r.append('chosen')
    if random:
        evaluate_c_r.append('random')

    if not evaluate_c_r:
        print("Choose at least chosen or random.")
        exit(1)

    for chosen_or_random in evaluate_c_r:
        for map_num in type_pdf:
            dir_name = generate_dir_name(chosen_or_random, map_num)
            full_dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(full_dir_path) and any(
                    os.path.isdir(os.path.join(full_dir_path, i)) for i in os.listdir(full_dir_path)):
                list_dir_names.append(full_dir_path)

    return list_dir_names


# ------------------------------------------------------------------------
# here the actual code starts

type_pdf_to_evaluate = [2, 3, 4, 5, 6, 7, 8]

# checking all the data, creating the correct combined folders with newest data
directories = list_dir_names(type_pdf_to_evaluate)

# print(directories)

# Create the JSON file
general_data = {}

for dir_name in directories:
    print(f"\nEvaluating dir {dir_name}\n")

    # Delete the "combined" subdirectory if it exists, to be sure no old data is used.
    combined_path = os.path.join(dir_name, "combined")
    if os.path.exists(combined_path):
        shutil.rmtree(combined_path)

    # Check the data is the same within the folder
    check_data_path(dir_name)

    # Sort the runs and make a combined folder
    sort_runs(dir_name)

    # After sorting runs, count the number of json files in "combined"
    num_json_files = len(glob.glob(f"{combined_path}/*.json"))
    # Store the result in general_data
    general_data[dir_name] = num_json_files

if os.path.exists(f"{base_dir}/general_data.json"):
    os.remove(f"{base_dir}/general_data.json")

# Save the result in a json file
with open(f"{base_dir}/general_data.json", "w") as json_file:
    json.dump(general_data, json_file, indent=4)

# ------------------------------------------------------------------------
# Preparing box_plots directory
box_plots_dir = f'{base_dir}/box_plots'
if os.path.exists(box_plots_dir):
    shutil.rmtree(box_plots_dir)
os.makedirs(box_plots_dir)

# Compare 'chosen' and 'random' for each type of PDF, make a box_plot if possible
for type_pdf in type_pdf_to_evaluate:
    print(f"Type box plot: {type_pdf}")
    dir_chosen_list = list_dir_names(type_pdf, chosen=True, random=False)
    dir_random_list = list_dir_names(type_pdf, chosen=False, random=True)
    if dir_chosen_list and dir_random_list:  # If both 'chosen' and 'random' directories exist for the current PDF type
        dir_chosen = dir_chosen_list[0]
        dir_random = dir_random_list[0]
        if os.path.exists(dir_chosen) and os.path.exists(dir_random):
            # Call compare function
            output_path = f"{box_plots_dir}/box_plot_type_{type_pdf}"
            compare_random_chosen(dir_random, dir_chosen, output_path, show_plot=False)
        else:
            print(f"Cannot compare type {type_pdf} because 'combined' directory does not exist.")
    else:
        print(f"Cannot compare type {type_pdf} because either 'chosen' or 'random' directory does not exist.")

# ------------------------------------------------------------------------
# Returning all the z_mesh plots in a folder
pdfs_plots_dir = f'{base_dir}/pdfs_plots'
if os.path.exists(pdfs_plots_dir):
    shutil.rmtree(pdfs_plots_dir)
os.makedirs(pdfs_plots_dir)

for type_pdf in type_pdf_to_evaluate:
    # Find directories for both 'chosen' and 'random'
    dir_list = list_dir_names(type_pdf)
    # Check if the dir_list is empty
    if dir_list:
        # Loop over the directories
        for dir_name in dir_list:
            # Construct the path to the JSON file
            json_file_path = os.path.join(dir_name, "global_mesh_data.json")
            # If the JSON file exists
            if os.path.isfile(json_file_path):
                # Prepare the output path
                output_path = f"{pdfs_plots_dir}/pdf_plot_type_{type_pdf}"
                # Call plot_z_mesh function
                plot_z_mesh(json_file_path, output_path)
                # Stop the loop if the JSON file is found
                break
            else:
                print(f"The file global_mesh_data.json does not exist in {dir_name}.")
    else:
        print(f"No directories exist for type {type_pdf}.")

# ------------------------------------------------------------------------
# Returning all the end pos plots with scale
print("\n\nStarting returning end pos with scale\n")
end_positions_dir = f'{base_dir}/end_positions_relative'
if os.path.exists(end_positions_dir):
    shutil.rmtree(end_positions_dir)
os.makedirs(end_positions_dir)

for type_pdf in type_pdf_to_evaluate:
    dir_random_list = list_dir_names(type_pdf, chosen=False, random=True)
    if dir_random_list:
        dir_random = dir_random_list[0]
        combined_dir = os.path.join(dir_random, "combined")
        if os.path.exists(combined_dir):
            # Prepare the output path
            output_path = f"{end_positions_dir}/end_positions_relative_type_{type_pdf}"
            # Call plot_end_pos function
            plot_end_pos(dir_random, output_path)
        else:
            print(f"'combined' directory does not exist for {dir_random}.")
    else:
        print(f"Random directory does not exist for type {type_pdf}.")

# ------------------------------------------------------------------------
# lloyds comparison
print("\n\nStarting lloyds comparison\n")
lloyds_dir = f'{base_dir}/lloyds_comparison'
if os.path.exists(lloyds_dir):
    shutil.rmtree(lloyds_dir)
os.makedirs(lloyds_dir)

n_runs_w_lloyds_dict = {}

for type_pdf in type_pdf_to_evaluate:
    dir_random_list = list_dir_names(type_pdf, chosen=False, random=True)
    if dir_random_list:
        dir_random = dir_random_list[0]
        combined_dir = os.path.join(dir_random, "combined")
        if os.path.exists(combined_dir):
            output_path = f"{lloyds_dir}/lloyds_data_type_{type_pdf}"
            n_runs_w_lloyds = get_lloyds_data_from_type(dir_random, output_path)
        else:
            print(f"'combined' directory does not exist for {dir_random}.")
    else:
        print(f"Random directory does not exist for type {type_pdf}.")

    print(f"Type {type_pdf} has {n_runs_w_lloyds} runs with lloyds data.")
    n_runs_w_lloyds_dict[type_pdf] = n_runs_w_lloyds

# ------------------------------------------------------------------------
# random vs chosen comparison
print("\n\nStarting random vs chosen comparison\n")
r_vs_c_dir = f'{base_dir}/random_vs_chosen'
if os.path.exists(r_vs_c_dir):
    shutil.rmtree(r_vs_c_dir)
os.makedirs(r_vs_c_dir)

for type_pdf in type_pdf_to_evaluate:
    dir_chosen_list = list_dir_names(type_pdf, chosen=True, random=False)
    dir_random_list = list_dir_names(type_pdf, chosen=False, random=True)
    if dir_chosen_list and dir_random_list:
        dir_chosen = dir_chosen_list[0]
        dir_random = dir_random_list[0]
        if os.path.exists(dir_chosen) and os.path.exists(dir_random):
            pass

# ------------------------------------------------------------------------
# generating latex data
print("\n\nStart exporting data to latex\n")
latex_dir = f'{base_dir}/latex_text'
if os.path.exists(latex_dir):
    shutil.rmtree(latex_dir)
os.makedirs(latex_dir)

generating_loyds_table(lloyds_dir, latex_dir)
generating_loyds_table(lloyds_dir, latex_dir, compact=True)


