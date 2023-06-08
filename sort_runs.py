import os
import json
import shutil

# not of importance to us, so made with chat GPT!

# The root directory that holds the 'global_dir...' directories
root_dir = 'combined_global_dirs_chosen_starts_3peaks_diff_sigma'

# Get a list of all 'global_dir...' directories
subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Collect the 'mw_vor_start_time' from each file along with its path
file_times = []
for subdir in subdirs:
    json_files = [f for f in os.listdir(subdir) if
                  f.endswith('.json') and f != 'global_mesh_data.json' and f != 'result_time_reduction.json']
    for json_file in json_files:
        file_path = os.path.join(subdir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            mw_vor_start_time = data['mw_vor_end_time']  # Assume the key always exists
            file_times.append((mw_vor_start_time, file_path, data))

# Sort the files by time
file_times.sort()

# Create new directory for combined files
combined_dir = os.path.join(root_dir, 'combined')
os.makedirs(combined_dir, exist_ok=True)

# Rename, modify, and move the files to the new directory
for i, (time, file_path, data) in enumerate(file_times):
    new_file_path = os.path.join(combined_dir, f'data_index_{i}.json')
    # Append the original file path to the data
    data['original_file_path'] = file_path
    # Write the modified data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)  # The indent parameter is optional. It makes the JSON file easier to read.
    # Move the modified file to the new directory
    shutil.copy(file_path, new_file_path)
