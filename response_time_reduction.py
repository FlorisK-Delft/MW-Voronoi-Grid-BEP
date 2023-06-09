import os
import json
import matplotlib.pyplot as plt
import numpy as np

root_dir = 'combined_global_dirs_run30mei_3punten_zelfde_groote'
map_with_json_files = f'{root_dir}/combined'

# if not os.path.exists(map_with_json_files):
#     os.makedirs(map_with_json_files)

json_files = [f for f in os.listdir(map_with_json_files) if f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']

count_mw_faster = 0
count_lloyds_faster = 0

total_start_time = 0.
total_mw_vor_end_time = 0.
total_lloyds_end_time_as_mw = 0.

start_time_avg_over_time = []
mw_vor_end_time_avg_over_time = []
lloyds_end_time_as_mw_avg_over_time = []

n_data = 0.0

for json_file in json_files:
    with open(f'{map_with_json_files}/{json_file}', 'r') as f:
        data = json.load(f)

    start_time = data["mw_vor_start_time"]
    mw_vor_end_time = data["mw_vor_end_time"]
    lloyds_end_time_as_mw = data["lloyds_end_time_as_mw"]

    if mw_vor_end_time < lloyds_end_time_as_mw:
        count_mw_faster += 1
    else:
        count_lloyds_faster += 1

    total_start_time += start_time
    total_mw_vor_end_time += mw_vor_end_time
    total_lloyds_end_time_as_mw += lloyds_end_time_as_mw

    n_data += 1.0

    start_time_avg_over_time.append(total_start_time / n_data)
    mw_vor_end_time_avg_over_time.append(total_mw_vor_end_time / n_data)
    lloyds_end_time_as_mw_avg_over_time.append(total_lloyds_end_time_as_mw / n_data)

avg_start_time = total_start_time / n_data
avg_mw_vor_end_time = total_mw_vor_end_time / n_data
avg_lloyds_end_time_as_mw = total_lloyds_end_time_as_mw / n_data

reduct_mw_percent = ((avg_start_time - avg_mw_vor_end_time) * 100 / avg_start_time)
reduction_lloyds_percent = ((avg_start_time - avg_lloyds_end_time_as_mw) * 100 / avg_start_time)

mw_faser_percent = (count_mw_faster * 100 / n_data)
# print the results
print(
    f"Average start time: {round(avg_start_time, 4)}"
    f"\nAverage mw vor end time: {round(avg_mw_vor_end_time, 4)}, "
    f"reduction of {round(reduct_mw_percent, 2)}%"
    f"\nAverage lloyds end time: {round(avg_lloyds_end_time_as_mw, 4)}, "
    f"reduction of {round(reduction_lloyds_percent , 2)}%"
    f"\n"
    f"\nIn {count_mw_faster} of {n_data} the mw voronoi resulted in a faster response time."
    f"\nEqual to {round(mw_faser_percent, 1)}% of cases"
)

# save the results in a json file
data = {
    "avg_start_time": avg_start_time,
    "avg_mw_vor_end_time": avg_mw_vor_end_time,
    "avg_lloyds_end_time_as_mw": avg_lloyds_end_time_as_mw,
    "reduct_mw_percent": reduct_mw_percent,
    "reduction_lloyds_percent": reduction_lloyds_percent,
    "count_mw_faster": count_mw_faster,
    "count_lloyds_faster": count_lloyds_faster,
    "number_of_data": n_data,
    "mw_faser_in_percent_cases" : mw_faser_percent

}
json_file_name = f'{map_with_json_files}/result_time_reduction.json'
with open(json_file_name, 'w') as file:
    json.dump(data, file, indent=4)

# make a plot how the average value converges
# At the end of your script, plot the results
plt.figure(figsize=(10, 6))
# plt.plot(start_time_avg_over_time, label='Start Time Avg')
plt.plot(mw_vor_end_time_avg_over_time, label='MW Voronoi End Time Avg')
# plt.plot(lloyds_end_time_as_mw_avg_over_time, label='Lloyd\'s End Time as MW Avg')
plt.xlabel('Step')
plt.ylabel('Average Time')
plt.title('Average Times Over Data Points')
plt.legend()
plt.grid(True)
plt.show()

# make a second plot, this time relative to the final position
final_avg_start_time = start_time_avg_over_time[-1]
final_avg_mw_vor_end_time = mw_vor_end_time_avg_over_time[-1]
final_avg_lloyds_end_time_as_mw = lloyds_end_time_as_mw_avg_over_time[-1]

relative_avg_start_time = [x - final_avg_start_time for x in start_time_avg_over_time]
relative_avg_mw_vor_end_time = [x - final_avg_mw_vor_end_time for x in mw_vor_end_time_avg_over_time]
relative_avg_lloyds_end_time_as_mw = [x - final_avg_lloyds_end_time_as_mw for x in lloyds_end_time_as_mw_avg_over_time]

plt.figure(figsize=(10, 6))
# plt.plot(relative_avg_start_time, label='Start Time Avg')
plt.plot(relative_avg_mw_vor_end_time, label='MW Voronoi End Time Avg')
# plt.plot(relative_avg_lloyds_end_time_as_mw, label='Lloyd\'s End Time as MW Avg')
plt.xlabel('Step')
plt.ylabel('Relative Average Time')
plt.title('Relative Average Times Over Steps')
plt.legend()
plt.grid(True)
plt.show()