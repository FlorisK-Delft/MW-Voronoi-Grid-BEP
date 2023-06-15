import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
import textwrap
from scipy import stats
import shutil

prefix_dict_2 = {
    1: '2 eclipse',
    2: 'Unimodal',
    3: 'Bimodal\\\\equal',
    4: 'Trimodal\\\\equal',
    5: 'Bimodal\\\\unequal height',
    6: 'Trimodal\\\\unequal height',
    7: 'Trimodal\\\\unequal sigma',
    8: 'Bimodal\\\\unequal sigma',
    9: '4-modal equal',
    10: '4-modal unequal height',
    11: '4-modal unequal sigma'
}

# dir_random_start = "data/3unequal_height_random_type6"
# dir_chosen_start = "data/3unequal_height_chosen_type6"

def calculate_statistics(values):
    return {
        "count (n-runs)": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std_dev": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "Q1": np.percentile(values, 25),
        "Q3": np.percentile(values, 75),
        "IQR": np.percentile(values, 75) - np.percentile(values, 25)
    }

def perform_t_test(list1, list2):
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(list1, list2, equal_var=False)

    # Print the results
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    return t_stat, p_value


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


def compare_random_chosen(dir_random_start, dir_chosen_start, output_path, show_plot=False):
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

    list_mw_vor_start_time_random, list_mw_vor_end_time_random, list_number_iterations_random, number_of_runs_random = return_list_of_data(
        dir_random_start)
    list_mw_vor_start_time_chosen, list_mw_vor_end_time_chosen, list_number_iterations_chosen, number_of_runs_chosen = return_list_of_data(
        dir_chosen_start)

    random_mean = np.mean(list_mw_vor_end_time_random)
    chosen_mean = np.mean(list_mw_vor_end_time_chosen)
    mean_decrease = ((random_mean - chosen_mean) / random_mean) * 100

    t_stat, p_value = perform_t_test(list_mw_vor_end_time_random, list_mw_vor_end_time_chosen)

    # print(list_mw_vor_end_time_chosen)
    # print(list_mw_vor_end_time_random)

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

    labels = [f'Random start positions, \n$n runs={number_of_runs_random}$',
              f'Chosen start positions, \n$n runs={number_of_runs_chosen}$']
    ax.set_xticklabels(labels, fontproperties=font_prop_regular, fontsize=16)

    ax.set_ylabel('Cost time\u00B2', fontproperties=font_prop_regular, fontsize=16)

    # title = 'Comparison of average response time²\nof final distribution'  # nog mee nemen hier tT met een streepje
    # wrapped_title = textwrap.fill(title, 37)  # Adjust the line width as needed
    # ax.set_title(wrapped_title, fontproperties=font_prop_bold, fontsize=20)
    annotation = f'p = {p_value:.2e}'

    annotation_text = f"{annotation}, cost mean \u2193: {mean_decrease:.1f}%"
    ax.annotate(annotation_text,
                xy=(0.95, 0.99),  # (x,y) coordinates where the annotation should be placed (relative to the axis)
                xycoords='axes fraction',  # indicates that the coordinates are fractions of the axis (0-1)
                fontsize=16,
                fontproperties=font_prop_regular,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black',
                          boxstyle='square,pad=0.5'))  # the bbox argument allows you to add a box around the text

    plt.savefig(f"{output_path}.png", dpi=300)  # dpi is the resolution of each png

    if show_plot is True:
        plt.show()

    output_data = {
        "random_statistics": calculate_statistics(list_mw_vor_end_time_random),
        "chosen_statistics": calculate_statistics(list_mw_vor_end_time_chosen),
        "mean_decrease": mean_decrease,
        "t_statistic": t_stat,
        "p_value": p_value
    }

    with open(f"{output_path}_data.json", 'w') as f:
        json.dump(output_data, f, indent=4)

    plt.close()

def r_and_c_to_latex(dir_files, output_dir, compact=False):
    sort_order = [2, 3, 5, 8, 4, 6, 7]

    # Process each JSON file in the directory
    json_files = [f for f in os.listdir(dir_files) if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

    # Sort the files according to the list
    json_files = sorted(json_files, key=lambda f: sort_order.index(int(f.split('_')[3].split('.')[0])) if int(
        f.split('_')[3].split('.')[0]) in sort_order else len(sort_order), reverse=False)

    # Create variable strings
    if compact:
        A = "Mean"
        B = "Median"
        C = "\\makecell[b]{Standard\\\\deviation}"
        D = "n runs"
        E = A
        F = B
        G = C
        H = D
        I = "Mean $\downarrow$"
        J = "Median $\downarrow$"
        K = "p value"
        output_txt = f'{output_dir}/table_r_c_compact.txt'
    else:
        A = "Mean"
        B = "Median"
        C = "\\makecell[b]{Standard\\\\deviation}"
        D = "n runs"
        E = A
        F = B
        G = C
        H = D
        I = "Mean $\downarrow$"
        J = "Median $\downarrow$"
        K = "p value"
        output_txt = f'{output_dir}/table_r_c_compact.txt'
        output_txt = f'{output_dir}/table_r_c.txt'

    # Open the output text file in write mode
    with open(output_txt, 'w') as outfile:
        # Write the table preamble
        outfile.write("\\begin{table}[h]\n")
        outfile.write("\\centering\n")
        if compact:
            outfile.write("\\begin{NiceTabular}{cccc}\n")
            outfile.write("\\toprule\n")
            outfile.write("& \\multicolumn{1}{c}{\\textbf{Random}} & \\multicolumn{1}{c}{\\textbf{Chosen}} & \\multicolumn{1}{c}{\\textbf{Result}}\\\\\n")
            outfile.write("\\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\cmidrule(lr){4-4}\n")
            outfile.write(f"\\textbf{{PDF Type}} & {{{A}}} & {{{E}}} & {{{I}}} \\\\\n")
        else:
            outfile.write("\\begin{NiceTabular}{cccccccccccc}\n")
            outfile.write("\\toprule\n")
            outfile.write("& \\multicolumn{4}{c}{\\textbf{Random start positions}} & \\multicolumn{4}{c}{\\textbf{Chosen start positions}} & \\multicolumn{3}{c}{\\textbf{Result}}\\\\\n")
            outfile.write("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-12}\n")
            outfile.write(f"\\textbf{{PDF Type}} & {{{A}}} & {{{B}}} & {{{C}}} & {{{D}}} & {{{E}}} & {{{F}}} & {{{G}}} & {{{H}}} & {{{I}}} & {{{J}}} & {{{K}}} \\\\\n")
        outfile.write("\\midrule\n")

        # Iterate over every row
        for i, json_file in enumerate(json_files):
            json_file_dir = f'{dir_files}/{json_file}'
            with open(json_file_dir, 'r') as f:
                data = json.load(f)

            type_of_pdf = int(json_file.split('_')[3].split('.')[0])
            row_name = prefix_dict_2[type_of_pdf]

            A_value = "{:.3f}".format(data['random_statistics']['mean'])
            B_value = "{:.3f}".format(data['random_statistics']['median'])
            C_value = "{:.2e}".format(data['random_statistics']['std_dev'])
            D_value = int(data['random_statistics']['count (n-runs)'])
            E_value = "{:.3f}".format(data['chosen_statistics']['mean'])
            F_value = "{:.3f}".format(data['chosen_statistics']['median'])
            G_value = "{:.2e}".format(data['chosen_statistics']['std_dev'])
            H_value = int(data['chosen_statistics']['count (n-runs)'])
            I_value = "{:.1f}".format(data['mean_decrease']) + '\%'
            J_value = "{:.1f}".format(((data['random_statistics']['median'] - data['chosen_statistics']['median']) /
                                        data['random_statistics']['median']) * 100) + '\%'
            K_value = "{:.2e}".format(data['p_value'])

            if compact:
                outfile.write(
                    f"\\makecell*{{{row_name}}} & {A_value} & {E_value} & {I_value} \\\\\n")
            else:
                outfile.write(
                    f"\\makecell*{{{row_name}}} & {A_value} & {B_value} & {C_value} & {D_value} & {E_value} & {F_value} & {G_value} & {H_value} & {I_value} & {J_value} & {K_value} \\\\\n")

        # Writing the end of the table
        outfile.write("\\bottomrule\n")
        outfile.write("\\end{NiceTabular}\n")
        outfile.write("\\end{table}\n")

        del outfile

# base_dir = "data"
# print("\n\nStart exporting data to latex\n")
# latex_dir = f'{base_dir}/latex_text'
# if not os.path.exists(latex_dir):
#     os.makedirs(latex_dir)
#
# box_plots_dir = f'{base_dir}/box_plots'
#
# r_and_c_to_latex(box_plots_dir, latex_dir)
#
# r_and_c_to_latex(box_plots_dir, latex_dir, compact=True)
