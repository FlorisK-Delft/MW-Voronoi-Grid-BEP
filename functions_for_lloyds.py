import os
import json
import shutil
import errno
import numpy as np
from scipy.stats import ttest_rel

prefix_dict = {
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

def remove_first_zero(string):
    if string.startswith('0.'):
        return string[1:]
    else:
        return string

def get_lloyds_data_from_type(dir_files, output):
    number_of_lloyds_runs = 0
    list_lloyds_end_time = []
    list_svmr_start_time = []
    list_svmr_end_time = []
    list_svmr_faster = []
    list_svmr_lloyds_reduction = []

    map_with_json_files = f'{dir_files}/combined'

    json_files = [f for f in os.listdir(map_with_json_files) if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

    for json_file in json_files:
        json_file_dir = f'{map_with_json_files}/{json_file}'
        with open(json_file_dir, 'r') as f:
            data = json.load(f)

        # Check if 'lloyds_end_time_as_mw' exists in the data
        if 'lloyds_end_time_as_mw' in data:
            mw_vor_start_time = float(data["mw_vor_start_time"])
            lloyds_start_time_as_mw = float(data["lloyds_start_time_as_mw"])

            # Calculate the percentage difference
            diff = abs((mw_vor_start_time - lloyds_start_time_as_mw) / mw_vor_start_time * 100)

            if diff > 1:
                continue

            number_of_lloyds_runs += 1

            svmr_start_time = float(data["mw_vor_start_time"])
            list_svmr_start_time.append(svmr_start_time)

            svmr_end_time = float(data["mw_vor_end_time"])
            list_svmr_end_time.append(svmr_end_time)

            lloyds_end_time = float(data["lloyds_end_time_as_mw"])
            list_lloyds_end_time.append(lloyds_end_time)

            # Check if SVMR was faster than Lloyd's
            list_svmr_faster.append(1 if svmr_end_time < lloyds_end_time else 0)

            svmr_lloyds_reduction = (lloyds_end_time - svmr_end_time) / lloyds_end_time * 100

            list_svmr_lloyds_reduction.append(svmr_lloyds_reduction)

    if not list_lloyds_end_time or not list_svmr_start_time or not list_svmr_end_time:
        output_data = {
            "lloyds_end_time": [],
            "svmr_start_time": [],
            "svmr_end_time": [],
            "svmr_faster": [],
            "svmr_faster_percentage": None,
            "average_svmr_start_time": None,
            "average_lloyds_end_time": None,
            "average_svmr_end_time": None,
            "svmr_decrease_percentage_compared_to_lloyds": None,
            "reduction_lloyds_compared_to_start": None,
            "reduction_svmr_compared_to_start": None,
            "number_of_lloyds_runs": 0,
            "svmr_lloyds_reduction_sigma": None,
            "t_statistic": None,
            "p_value": None
        }
    else:
        svmr_faster_percentage = round(sum(list_svmr_faster) / len(list_svmr_faster) * 100, 2)
        if svmr_faster_percentage < 100.0:
            print(f"The percentage is lower than 100: {svmr_faster_percentage}")
        average_svmr_start_time = sum(list_svmr_start_time) / len(list_svmr_start_time)
        average_lloyds_end_time = sum(list_lloyds_end_time) / len(list_lloyds_end_time)
        average_svmr_end_time = sum(list_svmr_end_time) / len(list_svmr_end_time)
        svmr_decrease_percentage = round((average_lloyds_end_time - average_svmr_end_time) / average_lloyds_end_time * 100,
                                         2)

        average_svmr_lloyds_reduction = sum(list_svmr_lloyds_reduction) / len(list_svmr_lloyds_reduction)

        reduction_lloyds_compared_to_start = round(
            (average_svmr_start_time - average_lloyds_end_time) / average_svmr_start_time * 100, 2)
        reduction_svmr_compared_to_start = round(
            (average_svmr_start_time - average_svmr_end_time) / average_svmr_start_time * 100, 2)

        svmr_lloyds_reduction_sigma = np.std(list_svmr_lloyds_reduction)

        if len(list_svmr_end_time) != len(list_lloyds_end_time):
            print("Not same length for lloyds and svmr, something went wrong.")
            exit(1)

        t_statistic, p_value = ttest_rel(list_svmr_end_time, list_lloyds_end_time)

        print("\n\nPaired Samples T-Test")
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)

        # Interpret the results
        alpha = 0.05  # Significance level
        if p_value < alpha:
            print("There is a significant difference between SVMR end time and Lloyd's end time.")
        else:
            print("There is no significant difference between SVMR end time and Lloyd's end time.")




        # Once all files have been processed, save the results to a JSON file
        output_data = {
            "svmr_faster_percentage": svmr_faster_percentage,
            "average_svmr_start_time": round(average_svmr_start_time, 3),
            "average_lloyds_end_time": round(average_lloyds_end_time, 3),
            "average_svmr_end_time": round(average_svmr_end_time, 3),
            "svmr_decrease_percentage_compared_to_lloyds": svmr_decrease_percentage,
            "reduction_lloyds_compared_to_start": reduction_lloyds_compared_to_start,
            "reduction_svmr_compared_to_start": reduction_svmr_compared_to_start,
            "number_of_lloyds_runs": number_of_lloyds_runs,
            "svmr_lloyds_reduction_sigma": round(svmr_lloyds_reduction_sigma, 3),
            "average_svmr_lloyds_reduction": round(average_svmr_lloyds_reduction, 3),
            "t_statistic": t_statistic,
            "p_value": p_value,
            "lloyds_end_time": list_lloyds_end_time,
            "svmr_start_time": list_svmr_start_time,
            "svmr_end_time": list_svmr_end_time,
            "svmr_faster": list_svmr_faster
        }

    with open(f"{output}.json", 'w') as f:
        json.dump(output_data, f, indent=4)

    return number_of_lloyds_runs


def generating_loyds_table(dir_files, output_dir, compact=False):
    sort_order = [2, 3, 5, 8, 4, 6, 7]  # this is how they come row for row in the table

    # Process each JSON file in the directory
    json_files = [f for f in os.listdir(dir_files) if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

    # Sort the files according to the list
    json_files = sorted(json_files, key=lambda f: sort_order.index(int(f.split('_')[3].split('.')[0])) if int(
        f.split('_')[3].split('.')[0]) in sort_order else len(sort_order), reverse=False)

    # Create variable strings
    if compact:
        A = "\\makecell[b]{cost \\\\ $time^{2}$}"
        B = A
        # C = "reduction"
        D = A
        # E = C
        # F = "n-runs"
        G = "\\makecell[b]{NSMR\\\\improvement}"
        H = "\\textit{p} value"
        output_txt = f'{output_dir}/table_lloyds_compact.txt'
    else:
        A = "\\makecell[b]{cost \\\\ $time^{2}$}"
        B = A
        #C = "Reduction"
        C = "\\makecell[b]{Reduction\\\\from Start}"
        D = A
        E = C
        F = "n-runs"
        G = "\\makecell[b]{NSMR \\\\ vs Lloyds}"
        H = "\\textit{p}-value"
        output_txt = f'{output_dir}/table_lloyds.txt'

    # Open the output text file in write mode
    with open(output_txt, 'w') as outfile:
        # Write the table preamble
        outfile.write("\\begin{table}[h]\n")
        outfile.write("\\caption{Lloyd's performance compared to NSMR's performance}\n")
        outfile.write("\\centering\n")
        if compact:
            outfile.write("\\begin{adjustbox}{width=\\columnwidth,center}\n")
            outfile.write("\\begin{NiceTabular}{cccccc}\n")
            outfile.write("\\toprule\n")
            outfile.write("& \\multicolumn{1}{c}{\\textbf{Start pos}} & \\multicolumn{1}{c}{\\textbf{Lloyd's pos}} & \\multicolumn{1}{c}{\\textbf{NSMR}} & \\multicolumn{2}{c}{\\textbf{Result}}\\\\\n")
            outfile.write("\\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\cmidrule(lr){4-4} \\cmidrule(lr){5-6}\n")
            outfile.write(f"\\textbf{{PDF Type}} & {{{A}}} & {{{B}}} & {{{D}}} & {{{G}}} & {{{H}}} \\\\\n")
        else:
            outfile.write("\\begin{NiceTabular}{ccccccccc}\n")
            outfile.write("\\toprule\n")
            outfile.write("& \\multicolumn{1}{c}{\\textbf{Start pos}} & \\multicolumn{2}{c}{\\textbf{Lloyd's pos}} & \\multicolumn{2}{c}{\\textbf{NSMR}} & \\multicolumn{3}{c}{\\textbf{Result}}\\\\\n")
            outfile.write("\\cmidrule(lr){2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-9}\n")
            outfile.write(f"\\textbf{{PDF Type}} & {{{A}}} & {{{B}}} & {{{C}}} & {{{D}}} & {{{E}}} & {{{F}}} & {{{G}}} & {{{H}}} \\\\\n")
        outfile.write("\\midrule\n")

        # Iterate over every row
        for i, json_file in enumerate(json_files):
            json_file_dir = f'{dir_files}/{json_file}'
            with open(json_file_dir, 'r') as f:
                data = json.load(f)

            if data["number_of_lloyds_runs"] == 0:
                continue

            type_of_pdf = int(json_file.split('_')[3].split('.')[0])
            row_name = prefix_dict[type_of_pdf]

            def remove_first_zero(string):
                if string.startswith('0.'):
                    return string[1:]
                else:
                    return string

            A_value = ("{:.3f}".format(round(data["average_svmr_start_time"], 3)))
            B_value = ("{:.3f}".format(round(data["average_lloyds_end_time"], 3)))
            C_value = "{:.1f}".format(round(data["reduction_lloyds_compared_to_start"], 1)) + '\%'
            D_value = ("{:.3f}".format(round(data["average_svmr_end_time"], 3)))
            E_value = "{:.1f}".format(round(data["reduction_svmr_compared_to_start"], 1)) + '\%'
            F_value = int(data["number_of_lloyds_runs"])
            G_value = "{:.1f}".format(round(data["svmr_decrease_percentage_compared_to_lloyds"], 1)) + '\%'
            H_value = "{:.2e}".format(data["p_value"])

            # Writing the row data
            if compact:
                outfile.write(f"\\makecell*{{{row_name}}} & {A_value} & {B_value} & {D_value} & {G_value} & {H_value} \\\\\n")
            else:
                outfile.write(f"\\makecell*{{{row_name}}} & {A_value} & {B_value} & {C_value} & {D_value} & {E_value} & {F_value} & {G_value} & {H_value} \\\\\n")

        # Writing the end of the table
        outfile.write("\\bottomrule\n")
        outfile.write("\\end{NiceTabular}\n")
        if compact:
            outfile.write("\\end{adjustbox}\n")
        outfile.write("\\end{table}\n")

        del outfile




# base_dir = "data"
# print("\n\nStart exporting data to latex\n")
# latex_dir = f'{base_dir}/latex_text'
# if os.path.exists(latex_dir):
#     shutil.rmtree(latex_dir)
# os.makedirs(latex_dir)
#
# lloyds_dir = f'{base_dir}/lloyds_comparison'
#
# generating_loyds_table(lloyds_dir, latex_dir)
#
# generating_loyds_table(lloyds_dir, latex_dir, compact=True)

