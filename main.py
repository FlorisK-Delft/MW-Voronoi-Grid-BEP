import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

# to make the gif:
import imageio

import os
import datetime
from classes import Plane, Robots
from voronoi_mw import VoronoiMW
from export_final import create_combined_image, save_data, save_gif, plot_avg_response_time, plot_p_dot_list


# Create necessary directory
def create_directory(run_number=None):
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time in the desired way
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the file name with the desired format
    if run_number is not None:
        dir_files_name = f"run_{run_number}_{formatted_date}"
    else:
        dir_files_name = f"run_{formatted_date}"

    os.makedirs(dir_files_name, exist_ok=True)
    return dir_files_name


def initialize_plane():
    init_plane = Plane([0, 10], [0, 10])
    resolution_x = ((init_plane.x_max - init_plane.x_min) * 20 + 1)
    resolution_y = ((init_plane.y_max - init_plane.y_min) * 20 + 1)
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(*init_plane.return_x_limits(), resolution_x),
        np.linspace(*init_plane.return_y_limits(), resolution_y),
    )
    return init_plane, x_mesh, y_mesh


def initialize_gaussian(x_mesh, y_mesh, init_plane):
    x_center = (init_plane.x_max + init_plane.x_min) / 2
    y_center = (init_plane.y_max + init_plane.y_min) / 2
    x_sigma = (init_plane.x_max - init_plane.x_min) / 2
    y_sigma = (init_plane.y_max - init_plane.y_min) / 2

    z_mesh = gaussian_2d(x_mesh, y_mesh, x_center, y_center, x_sigma, y_sigma)
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh


def initialize_robots(init_plane, number_of_robots=5, init_speeds=None,
                      all_black=False, random_pos_bool=True, random_speed_bool=False, print_robots=False):
    if init_speeds is None:
        init_speeds = [3, 3, 2, 2, 1]
    if random_pos_bool:
        x_random = np.random.uniform(init_plane.x_min, init_plane.x_max, number_of_robots)
        y_random = np.random.uniform(init_plane.y_min, init_plane.y_max, number_of_robots)
        positions = np.column_stack((x_random, y_random))
        if random_speed_bool:
            speed_robots = [random.randint(1, 3) for _ in range(number_of_robots)]
        else:
            speed_robots = init_speeds[:number_of_robots]
    else:  # static
        positions = np.array([[2.5, 1.5], [1, 8], [8, 8], [8, 1]])
        speed_robots = init_speeds[:number_of_robots]
    init_robots = Robots(init_plane, positions, speed_robots)

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


# def gaussian_2d(x, y, x0, y0, xsig, ysig):
#     return np.exp(-10 * (((x - x0) / xsig) ** 2 + ((y - y0) / ysig) ** 2))


def gaussian_2d(x_mesh, y_mesh, x0, y0, xsig, ysig):
    xsig_top = 0.5 * xsig
    ysig_top = 1.5 * ysig
    x0 *= 0.5
    y0 *= 1.5
    x0_bottom = x0 * 1.5
    y0_bottom = y0 * 0.5
    xsig_bottom = 1.5 * xsig
    ysig_bottom = 0.5 * ysig

    top_quadrant = np.exp(-8 * (((x_mesh - x0) / xsig_top) ** 2 + ((y_mesh - y0) / ysig_top) ** 2))
    bottom_quadrant = (np.exp(-8 * (((x_mesh - x0_bottom) / xsig_bottom) ** 2
                                    + ((y_mesh - y0_bottom) / ysig_bottom) ** 2)))

    return top_quadrant + bottom_quadrant


def assign_robot2voronoi(x_mesh, y_mesh, z_mesh, robots_for_vor, init_voronois, avg_response_time_i=0):
    for index_i in range(x_mesh.shape[0]):
        for index_j in range(x_mesh.shape[1]):
            grid_coordinate = np.array([x_mesh[index_i, index_j], y_mesh[index_i, index_j]])
            time = 999999
            fastest_robot = 0
            for k in range(robots_for_vor.number_of_robots()):
                time_robot = np.linalg.norm(robots_for_vor.return_position(k) -
                                            grid_coordinate) / robots_for_vor.return_max_speed(k)
                if time_robot < time:
                    fastest_robot = k
                    time = time_robot

            avg_response_time_i += time ** 2 * z_mesh[index_i, index_j]

            init_voronois[fastest_robot].add_grid_point(grid_coordinate,
                                                        float(z_mesh[index_i, index_j]), index_i, index_j)

    return avg_response_time_i


def get_border_voronoi(current_voronois, z_mesh):
    # start with making a grid the exact same size as z, but filled with zeros
    total_grid = np.zeros_like(z_mesh)

    # iterate for every voronoi
    for voronoi_number, voronoi in enumerate(current_voronois):
        array_i = np.copy(voronoi.grid_coordinates)

        # for every index in vor make it equal to the number of which voronoi it is,
        # so the first plane of voronoi 1 gets filled with only 1's, second one with only 2's, etc
        for j in range(len(array_i)):
            total_grid[voronoi.index_x[j], voronoi.index_y[j]] = voronoi_number + 1

    # Because every voronoi has a different number for the plane there exist a gradient between those planes,
    # calculate what the gradient is and the if any gradient exist make it equal to 1, otherwise make 'almost' 0 = 0
    grad = np.gradient(total_grid)
    grad = np.where(np.isclose(grad, 0, atol=1e-8), 0, 1)

    # adds the gradient in y and in x direction together so a border forms when plotted
    border = grad[0] + grad[1]

    # makes sure all values are ether 0 or 1 so the border can be plotted with a homogenous colour
    border = np.where(np.isclose(border, 0, atol=1e-8), 0, 1)

    # print(border)
    return border


def plot_current(iteration, robots_class, z_mesh, current_voronois,
                 plane_class, colors_robots_list, images_list, dir_files_name,
                 arrow_scale_var=1, always_show=False, last_iteration=False,
                 show_every_n_times=30):
    # def plot_current(iteration, last_iteration=False):
    plt.clf()  # clears the previous picture, so not all 15 appear at once
    plt.figure()

    # plot the robot positions
    plt.scatter(*zip(*robots_class.positions), c=colors_robots_list)

    d_letter = 0.22
    for index_pos in range(len(robots_class.robot_positions())):
        # plot the robot speed for every robot
        plt.text(robots_class.return_position(index_pos)[0] + d_letter,
                 robots_class.return_position(index_pos)[1] - d_letter,
                 f"${int(robots_class.return_max_speed(index_pos))}$",
                 fontsize=9)  # maybe in the text: v{index_pos + 1} =

    # plot the gradient of the pdf (this is z)
    plt.imshow(z_mesh, origin='lower',
               extent=(plane_class.x_min, plane_class.x_max, plane_class.y_min, plane_class.y_max),
               alpha=0.5)
    plt.colorbar()

    # plot the center point of mass
    for voronoi in current_voronois:
        plt.plot(*voronoi.center_of_mass(), 'x', c="blue")

    # plot the arrow of the gradient descent, with arrow_scale as variable
    for voronoi in current_voronois:
        px, py = arrow_scale_var * voronoi.gradient_descent()
        robot_x, robot_y = voronoi.position()

        length = np.sqrt(px ** 2 + py ** 2)

        # plt.quiver(robot_x, robot_y, px, py, angles='xy', scale_units='xy', scale=3, color='r')
        plt.arrow(
            robot_x, robot_y, px, py,
            color='red',
            width=min(length / 220, 0.05),
            head_width=min(length / 4, 0.2),
            length_includes_head=True
        )

    # plot the voronoi boundary's
    cmap = colors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(get_border_voronoi(current_voronois, z_mesh), cmap=cmap, norm=norm,
               extent=(plane_class.x_min, plane_class.x_max, plane_class.y_min, plane_class.y_max),
               origin='lower')

    # saves the iteration in the folder, and append to images for the gif
    # save the plot as png
    filename = f'{dir_files_name}/{iteration}.png'

    # save the png with the title
    plt.title(f'{iteration}')
    plt.savefig(filename, dpi=150)  # dpi is the resolution of each png

    # read the png image file and append it to the list, without title
    images_list.append(imageio.v3.imread(filename))

    # show the total plot
    if (iteration % show_every_n_times) == 0:
        plt.show()
    elif always_show:
        plt.show()
    elif last_iteration:
        plt.show()
    else:
        plt.close()
        # plt.clf()

    return images_list


for test in range(3):
    # ---------------------------------------------------------------------------------------------------------------------
    dir_files = create_directory(test)

    # Create the mesh grid
    plane, x, y = initialize_plane()

    z = initialize_gaussian(x, y, plane)

    # choose random or static for testing:
    images = []

    robots, colors_robots = initialize_robots(plane, all_black=True)

    # ---------------------------------------------------------------------------------------------------------------------
    # These are the most important variables to set!
    dt = 0.4
    iterations = 800
    stop_criterion = 0.005
    p_dot_max = None
    arrow_scale = 6
    # ---------------------------------------------------------------------------------------------------------------------
    avg_response_time = []
    p_dot_list = []
    quickest_response_time = 999999

    for i in range(iterations):
        print(f"\nCalculating iteration {i} \nCurrent p_dot_max: {p_dot_max}, stopping if p_dot_max < {stop_criterion}")

        # creates (for now empty) voronois for every robot
        voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                    range(robots.number_of_robots())]

        avg_response_time.append(assign_robot2voronoi(x, y, z, robots, voronois, avg_response_time_i=0))

        # this plot is with the current position and current location
        # plot_current(i,,

        images = plot_current(i, robots, z, voronois, plane, colors_robots, images, dir_files,
                              arrow_scale_var=arrow_scale, last_iteration=False)

        # calculate the next positions, return the maximum robot displacement p_dot to look for stop criterion
        p_dot_list_i, p_dot_max = robots.time_step_all(voronois, dt)

        p_dot_list.append(p_dot_list_i)

        if avg_response_time[i] < quickest_response_time:
            quickest_response_time = avg_response_time[i]
            print(f"Current average response time: {round(avg_response_time[i], 4)} (This time is the quickest)")
        else:
            print(f"Quickest response time: {round(quickest_response_time, 4)},"
                  f"\nCurrent response time: {round(avg_response_time[i], 4)} (Current time is NOT the quickest)")

        # stop criterion, if the vector for moving the robots gets small enough, stop moving the robots.
        if p_dot_max < stop_criterion:
            print(
                f"\nThe max p dot ({p_dot_max}) if smaller than {stop_criterion}, "
                f"iteration stopped. \nStopped at iteration: {i}")

            # makes sure the final plot get shown
            voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                        range(robots.number_of_robots())]

            avg_response_time.append(assign_robot2voronoi(x, y, z, robots, voronois, avg_response_time_i=0))

            # important, i+1 iteration!, robots have moved, this is thus actually the next iteration
            images = plot_current(i + 1, robots, z, voronois, plane, colors_robots, images, dir_files,
                                  arrow_scale_var=arrow_scale, last_iteration=True)

            p_dot_list_temp = [np.linalg.norm(voronois[i].gradient_descent()) for i in range(len(voronois))]
            # add the last velocity vectors for the plot, actually not using the time step:
            p_dot_list.append(p_dot_list_temp)

            break

        # deletes the voronois, so it can be used again next iteration
        del voronois

    plot_avg_response_time(avg_response_time, dir_files)

    plot_avg_response_time(avg_response_time, dir_files, log=True)

    # show the velocity vector of all the robots over time
    plot_p_dot_list(p_dot_list, stop_criterion, dir_files)

    print(
        f"The average response time\u00B2 at the start was: {avg_response_time[0]}."
        f"\nThe average response time\u00B2 at the end was:  {avg_response_time[-1]}"
        f"\nThe algorithm reduced the response time\u00B2 by "
        f"{round((avg_response_time[0] - avg_response_time[-1]) / (avg_response_time[0]), 3) * 100}%"
    )

    # create the gif:
    save_gif(images, dir_files)

    # save avg_response_time and p_dot_list to csv
    save_data(robots, avg_response_time, p_dot_list, dir_files)

    # create an overview of the most important data. So a test can be analysed quickly
    create_combined_image(
        start_index_png=f'{dir_files}/{0}.png',
        end_index_png=f'{dir_files}/{len(avg_response_time) - 1}.png',
        average_response_time_png=f"{dir_files}/Avg_response_time.png",
        velocity_robots_png=f"{dir_files}/Velocity_robots.png",
        start_time=avg_response_time[0],
        end_time=avg_response_time[-1],
        robot_info_list=robots.robot_p_and_v_array(),
        output_path=f"{dir_files}/data_overview_{datetime.datetime.now()}.png"
    )
