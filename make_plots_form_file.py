import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import shutil
import datetime

from classes import Plane, Robots
from voronoi_mw import VoronoiMW, assign_robot2voronoi, get_border_voronoi, response_time_mw_voronoi

now = datetime.datetime.now()
formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

output_path = 'data/created_images'
if os.path.exists(output_path):
    # shutil.rmtree(output_path)
    pass
else:
    os.makedirs(output_path)

# set the font
font_path_regular = './lmroman7-regular.otf'
font_prop_regular = fm.FontProperties(fname=font_path_regular)

font_path_bold = './lmroman7-bold.otf'
font_prop_bold = fm.FontProperties(fname=font_path_bold)

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

def initialize_robots(init_plane, number_of_robots=None, init_speeds=None, start_positions=None,
                      all_black=False, print_robots=False, min_speed=1, max_speed=3):
    if number_of_robots is None:
        number_of_robots = len(start_positions)
        if number_of_robots == 0:
            number_of_robots = 5
    if init_speeds is None:
        exit(1)
    else:
        speed_robots = init_speeds

    if start_positions is None:
        x_random = np.random.uniform(init_plane.x_min, init_plane.x_max, number_of_robots)
        y_random = np.random.uniform(init_plane.y_min, init_plane.y_max, number_of_robots)
        positions_r = np.column_stack((x_random, y_random))
    else:
        positions_r = start_positions

    init_robots = Robots(init_plane, positions_r, speed_robots)

    if print_robots:
        print(
            np.array(["x", "y", "v"]),
            "\n",
            init_robots.robot_p_and_v_array()
        )

    # makes sure there are the same amount of random colours as there are robots
    init_colors_robots = []
    if all_black:
        for robot in range(init_robots.number_of_robots()):
            init_colors_robots.append("black")
    else:
        all_colors = list(colors.CSS4_COLORS.values())
        np.random.shuffle(all_colors)
        init_colors_robots = all_colors[:init_robots.number_of_robots()]

    return init_robots, init_colors_robots


def generate_dir_name(chosen_or_random, fit_type):
    prefix = prefix_dict.get(fit_type, '')
    dir_name = f'{prefix}{chosen_or_random}_type{fit_type}'
    return dir_name


print(prefix_dict)
type_input = int(input("Which type do you want to plot?"))


index_json = input("What is the index of the json?")

bool_chosen_or_random = input("Do you want chosen or random? (Y fo Yes/chosen)")
if bool_chosen_or_random == "Y":
    chosen_or_random = "chosen"
else:
    chosen_or_random = "random"

if chosen_or_random == "random":
    bolean_ask = input("Want to show lloyds sim? (Y fo Yes)")
else:
    bolean_ask = "No"

if bolean_ask == "Y":
    loyds_sim = True
    plot_both_b = input("Plot SVMR voronois also? (Y for Yes)")
else:
    loyds_sim = False
    plot_both_b = None

colour_bar_ask = input("Plot the colour bar? (Y fo Yes)")
start_or_end_pos = input("For start pos (Y fo Yes), otherwise end pos will be plotted")

if colour_bar_ask == "Y":
    colour_bar = True
else:
    colour_bar = False

if start_or_end_pos == "Y":
    start_pos_plot = True
else:
    start_pos_plot = False

if plot_both_b == "Y":
    plot_both = True
else:
    plot_both = False

dir_folder = "data/" + generate_dir_name(chosen_or_random, type_input)
print("Currently looking in: " + dir_folder)
dir_json_file = dir_folder + f"/combined/data_index_{index_json}.json"

if not os.path.exists(dir_json_file):
    print(f"Chosen path {dir_json_file} does not exist.")
    exit(1)
else:
    print(f"File {dir_json_file} does exist, continue")

with open(dir_json_file, 'r') as f:
    data = json.load(f)

if start_pos_plot:
    # retreive the start positions of the plot we want to make
    start_positions_and_v = data["start_positions_and_v"]
elif not start_pos_plot and not loyds_sim:
    start_positions_and_v = data["mw_vor_end_positions_and_v"]
elif not start_pos_plot and loyds_sim:
    start_positions_and_v = data["lloyds_end_positions_and_v"]

speed_robots_init = [int(pos[-1]) for pos in start_positions_and_v]
positions = np.array([np.array(pos[:2]) for pos in start_positions_and_v])

print(speed_robots_init)
print(positions)


with open(f'{dir_folder}/global_mesh_data.json', 'r') as f:
    data = json.load(f)

# Get mesh data
z_mesh_list = data['z_mesh']
x_mesh_list = data['x_mesh']
y_mesh_list = data['y_mesh']

# Convert to numpy arrays
z_mesh = np.array(z_mesh_list)
x_mesh = np.array(x_mesh_list)
y_mesh = np.array(y_mesh_list)

# Get x and y bounds
x_min, x_max = np.min(x_mesh), np.max(x_mesh)
y_min, y_max = np.min(y_mesh), np.max(y_mesh)

arrow_scale = 6

plane = Plane([x_min, x_max], [y_min, y_max])

robots, colors_robots = initialize_robots(plane, start_positions=positions, init_speeds=speed_robots_init,
                                              all_black=True)

if loyds_sim:
    gain = 1
else:
    gain = 4

voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                    range(robots.number_of_robots())]

avg_time, avg_time_speed_eq = assign_robot2voronoi(x_mesh, y_mesh, z_mesh, robots, voronois,
                                                      loyds=loyds_sim)




# Create a new figure
plt.figure()

plt.scatter(*zip(*robots.positions), c=colors_robots)

d_letter = 0.22
for index_pos in range(len(robots.robot_positions())):
    # plot the robot speed for every robot
    plt.text(robots.return_position(index_pos)[0] + d_letter,
             robots.return_position(index_pos)[1] - d_letter,
             f"${int(robots.return_max_speed(index_pos))}$",
             fontsize=9)

if not plot_both:
    for voronoi in voronois:
        plt.plot(*voronoi.center_of_mass(), 'x', c="blue")

for voronoi in voronois:
    px, py = arrow_scale * voronoi.gradient_descent()
    robot_x, robot_y = voronoi.position()

    length = np.sqrt(px ** 2 + py ** 2)

    if not plot_both:
        # plt.quiver(robot_x, robot_y, px, py, angles='xy', scale_units='xy', scale=3, color='r')
        plt.arrow(
            robot_x, robot_y, px, py,
            color='red',
            width=min(length / 220, 0.05),
            head_width=min(length / 4, 0.2),
            length_includes_head=True
        )

if not loyds_sim:
    colours_base_border = [(0, 0, 0, 0), (0, 0, 0, 1)]
elif loyds_sim and not plot_both:
    colours_base_border = [(0, 0, 0, 0), (0, 0, 0, 1)]
elif loyds_sim and plot_both:
    colours_base_border = [(0, 0, 0, 0), (0, 0, 0, 0.5)]

# plot the voronoi boundary's
cmap = colors.ListedColormap(colours_base_border)
bounds = [0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(get_border_voronoi(voronois, z_mesh), cmap=cmap, norm=norm,
           extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max),
           origin='lower')

if loyds_sim and plot_both:
    voronois_2 = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                range(robots.number_of_robots())]
    avg_time, avg_time_speed_eq = assign_robot2voronoi(x_mesh, y_mesh, z_mesh, robots, voronois_2,
                                                       loyds=False)
    # plot the voronoi boundary's
    cmap = colors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(get_border_voronoi(voronois_2, z_mesh), cmap=cmap, norm=norm,
               extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max),
               origin='lower')

# Plot z_mesh data
plt.imshow(z_mesh, origin='lower', extent=(x_min, x_max, y_min, y_max), alpha=0.5)

if colour_bar:
    # Add a colorbar
    cbar = plt.colorbar(format=FormatStrFormatter('%.1e'))  # Use scientific notation with 2 decimal places

    # Create the font properties object with the specified font
    fontprops = fm.FontProperties(fname=font_path_regular, size=12)

    # Set the label for the colorbar
    string_density = "Normalized probability density"
    cbar.set_label(string_density, fontproperties=fontprops)

    # Set the font for the color bar labels
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontproperties(fontprops)

string_image = formatted_date
if loyds_sim:
    string_image += "_loyds"
else:
    string_image += "_svmr"

if start_pos_plot:
    string_image += "_start"
else:
    string_image += "_end"

# Save the figure
plt.savefig(f"{output_path}/"+ string_image + "_" + chosen_or_random + f"_type_{type_input}_index_{index_json}.png", dpi=300)  # dpi is the resolution of the output image

# Close the figure to free memory
plt.close()