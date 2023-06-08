import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

# to make the gif:
import imageio

import os
import datetime
from classes import Plane, Robots
from voronoi_mw import VoronoiMW, assign_robot2voronoi, get_border_voronoi, response_time_mw_voronoi
from export_final import create_combined_image, save_gif, plot_avg_response_time, plot_p_dot_list, compare_loyds_to_mw, export_data_run, append_lloyds_run_to_data, export_mesh
from probability_density_function import pdfunction
from starting_pos_func_v2_floris import distribute_robots_over_peaks

# Create necessary directory
def create_directory(run_number=None, loyds=False):
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time in the desired way
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the file name with the desired format
    if run_number is not None and (loyds is True):
        dir_files_name = f"run_{run_number}_{formatted_date}_loyds"
    elif run_number is not None and loyds is False:
        dir_files_name = f"run_{run_number}_{formatted_date}"
    elif run_number is None and loyds is True:
        dir_files_name = f"run_{formatted_date}_loyds"
    elif run_number is None and loyds is False:
        dir_files_name = f"run_{formatted_date}"

    os.makedirs(dir_files_name, exist_ok=True)
    return dir_files_name

def create_global_directory():
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    global_dir_name = f"global_dir_{formatted_date}"
    os.makedirs(global_dir_name, exist_ok=True)
    return global_dir_name

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


def initialize_robots(init_plane, number_of_robots=None, init_speeds=None, start_positions=None,
                      all_black=False, print_robots=False, min_speed=1, max_speed=3):
    if number_of_robots is None:
        number_of_robots = len(start_positions)
        if number_of_robots == 0:
            number_of_robots = 5
    if init_speeds is None:
        speed_robots = [random.randint(min_speed, max_speed) for _ in range(number_of_robots)]
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


def simulate_mw_voronoi(max_iterations, stop_criterion_simulation, plane, x, y, positions_sim=None,
                        speed_sim=None, dt_sim=0.1, arrow_scale_sim=1, loyds_sim=False, dir_json_file=None, type_pdf_sim=1):
    # ---------------------------------------------------------------------------------------------------------------------
    dir_files = create_directory(test, loyds=loyds_sim)

    # old way of defining z, still in tact:
    # z = initialize_gaussian(x, y, plane)

    # new way of defining z
    z = pdfunction(x, y, type = type_pdf_sim, sigma_x=5, sigma_y=5)

    # choose random or static for testing:
    images = []

    robots, colors_robots = initialize_robots(plane, start_positions=positions_sim, init_speeds=speed_sim,
                                              all_black=True)

    start_positions_to_save = (robots.robot_p_and_v_array()).tolist()


    if loyds_sim:
        gain = 1
    else:
        gain = 4

    avg_response_time = []
    p_dot_list = []
    p_dot_max = None

    if loyds_sim:
        avg_response_time_as_mw = []
        avg_response_time_speed_eq = []

    for i in range(max_iterations):
        print(
            f"\nCalculating iteration {i} \nCurrent p_dot_max: {p_dot_max}, stopping if p_dot_max < {stop_criterion_simulation}")

        # creates (for now empty) voronois for every robot
        voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                    range(robots.number_of_robots())]

        avg_time, avg_time_speed_eq = assign_robot2voronoi(x, y, z, robots, voronois,
                                                      loyds=loyds_sim)

        avg_response_time.append(avg_time)

        if loyds_sim:
            avg_response_time_as_mw.append(response_time_mw_voronoi(x, y, z, robots))
            avg_response_time_speed_eq.append(avg_time_speed_eq)

        # this plot is with the current position and current location
        # plot_current(i,,

        images = plot_current(i, robots, z, voronois, plane, colors_robots, images, dir_files,
                              arrow_scale_var=arrow_scale_sim, last_iteration=False)

        # calculate the next positions, return the maximum robot displacement p_dot to look for stop criterion
        p_dot_list_i, p_dot_max = robots.time_step_all(voronois, dt_sim, gain_p=gain, loyds_=loyds_sim)

        p_dot_list.append(p_dot_list_i)

        print(f"Current average response time: {round(avg_response_time[i], 4)}")
        if loyds_sim:
            print(f"Current average response time, as if MW vor: {round(avg_response_time_as_mw[i], 4)}"
                  f"\nCurrent average response time, as if equal speed: {round(avg_response_time_speed_eq[i], 4)}")

        # stop criterion, if the vector for moving the robots gets small enough, stop moving the robots.
        if p_dot_max < stop_criterion_simulation:
            print(
                f"\nThe max p dot ({p_dot_max}) if smaller than {stop_criterion_simulation}, "
                f"iteration stopped. \nStopped at iteration: {i}")

            # makes sure the final plot get shown
            voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                        range(robots.number_of_robots())]

            avg_time, avg_time_speed_eq = assign_robot2voronoi(x, y, z, robots, voronois,
                                                               loyds=loyds_sim)

            avg_response_time.append(avg_time)

            if loyds_sim:
                avg_response_time_as_mw.append(response_time_mw_voronoi(x, y, z, robots))
                avg_response_time_speed_eq.append(avg_time_speed_eq)

            # important, i+1 iteration!, robots have moved, this is thus actually the next iteration
            images = plot_current(i + 1, robots, z, voronois, plane, colors_robots, images, dir_files,
                                  arrow_scale_var=arrow_scale_sim, last_iteration=True)

            p_dot_list_temp = [np.linalg.norm(voronois[i].gradient_descent()) for i in range(len(voronois))]
            # add the last velocity vectors for the plot, actually not using the time step:
            p_dot_list.append(p_dot_list_temp)

            break

        # deletes the voronois, so it can be used again next iteration
        del voronois

    plot_avg_response_time(avg_response_time, dir_files)

    if loyds_sim:
        plot_avg_response_time(avg_response_time_as_mw, dir_files, title="Plot of average response time as if MW")
        plot_avg_response_time(avg_response_time_speed_eq, dir_files, title="Plot of average response time speeds eq")

    #plot_avg_response_time(avg_response_time, dir_files, log=True)

    # show the velocity vector of all the robots over time
    plot_p_dot_list(p_dot_list, stop_criterion, dir_files)

    print(
        f"The average response time\u00B2 at the start was: {avg_response_time[0]}."
        f"\nThe average response time\u00B2 at the end was:  {avg_response_time[-1]}"
        f"\nThe algorithm reduced the response time\u00B2 by "
        f"{round((avg_response_time[0] - avg_response_time[-1]) / (avg_response_time[0]) * 100, 2)}%"
    )

    if loyds_sim:
        print(
            f"\nLoyds:"
            f"\nThe avg_response time\u00B2 at the start, calculated as if loyds,"
            f" was: {avg_response_time_as_mw[0]}"
            f"\nThe avg_response time\u00B2 on the end, calculated as if loyds,"
            f" was: {avg_response_time_as_mw[-1]}"
        )

    # create the gif:
    save_gif(images, dir_files)

    # save avg_response_time and p_dot_list to csv
    # save_data(robots, avg_response_time, p_dot_list, dir_files)

    # for the windows students:
    now = datetime.datetime.now()
    formatted_date_combined = now.strftime("%Y-%m-%d_%H-%M-%S")

    # create an overview of the most important data. So a test can be analysed quickly
    create_combined_image(
        start_index_png=f'{dir_files}/{0}.png',
        end_index_png=f'{dir_files}/{len(avg_response_time) - 1}.png',
        average_response_time_png=f"{dir_files}/Avg_response_time.png",
        velocity_robots_png=f"{dir_files}/Velocity_robots.png",
        start_time=avg_response_time[0],
        end_time=avg_response_time[-1],
        robot_info_list=robots.robot_p_and_v_array(),
        output_path=f"{dir_files}/data_overview_{formatted_date_combined}.png"
    )

    if test == 0:
        export_mesh(
            x, y, z,
            output_path=global_dir
        )


    if loyds_sim:
        append_lloyds_run_to_data(dir_json_file,
                                  (robots.robot_p_and_v_array()).tolist(),
                                  avg_response_time,
                                  avg_response_time_as_mw,
                                  avg_response_time_speed_eq,
                                  gain
                                  )
        return avg_response_time, avg_response_time_as_mw, avg_response_time_speed_eq
    else:
        dir = export_data_run(
            start_positions_to_save,
            (robots.robot_p_and_v_array()).tolist(),
            stop_criterion_simulation,
            avg_response_time,
            p_dot_list,
            gain,
            dt,
            test,
            output_path=global_dir
        )
        return avg_response_time, 0, 0, dir


global_dir = create_global_directory()

for test in range(150):
    # Create the mesh grid
    plane, x, y = initialize_plane()

    # ---------------------------------------------------------------------------------------------------------------------
    # These are the most important variables to set!
    dt = 0.3  # the time step
    iterations = 1000  # the maximum number of iterations
    stop_criterion = 0.003 # if the fastest robot moves slower (p_dot) than the stop criterion the algorithm will break
    arrow_scale = 6  # to decide how the arrows should be shown

    # type 1 = original
    # type 2 = 1 gausian in the middel
    # type 3 = 2 gausian top left and bottom right
    # type 4 = 3 gausian, top left and right, bottom middle
    # type 5
    # type 6
    # type 7

    type_pdf = 7  # see list of what types in probability density function file

    number_of_robots = 12
    # speed_robots = [5, 5, 4, 4, 3, 2, 1]
    speed_robots_init = [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]

    random_start_pos = 2

    if random_start_pos == 1:
        x_random = np.random.uniform(plane.x_min, plane.x_max, number_of_robots)
        y_random = np.random.uniform(plane.y_min, plane.y_max, number_of_robots)
        positions_robots_start = np.column_stack((x_random, y_random))
        speed_robots = speed_robots_init
    elif random_start_pos == 2:
        z = pdfunction(x, y, type=type_pdf, sigma_x=5, sigma_y=5)

        positions_robots_start, speed_robots = distribute_robots_over_peaks(x, y, z, speed_list=speed_robots_init)
    else:
        positions = np.array([
            np.array([2.5,7.]),
            np.array([2.7,4.]),
            np.array([2.5,8.]),
            np.array([1.8,4.]),
            np.array([5.4,3.8])
        ])
        speed_robots = speed_robots_init

        positions_robots_start = positions


    # positions = np.array([[2.5, 1.5], [1, 8], [8, 8], [8, 1]])
    # ---------------------------------------------------------------------------------------------------------------------
    response_time, _, _ , dir_json = simulate_mw_voronoi(iterations, stop_criterion, plane, x, y,
                                                         positions_robots_start, speed_sim=speed_robots, dt_sim=dt,
                                                         arrow_scale_sim=6, loyds_sim=False, type_pdf_sim=type_pdf)
    # loyds_response_time, loyds_mw_voronoi_time, loyds_time_speed_eq = simulate_mw_voronoi(iterations, stop_criterion,
    #                                                                                       plane, x, y,
    #                                                                                       positions_robots_start,
    #                                                                                       speed_sim=speed_robots,
    #                                                                                       dt_sim=dt, arrow_scale_sim=6,
    #                                                                                       loyds_sim=True,
    #                                                                                       dir_json_file=dir_json,
    #                                                                                       type_pdf_sim=type_pdf)
    #
    # compare_loyds_to_mw(response_time, loyds_response_time, loyds_mw_voronoi_time, loyds_time_speed_eq, test)
    #
    # print(f"\nMW Vor Time\u00B2: {response_time[-1]}"
    #       f"\nLoyds vor Time\u00B2: {loyds_response_time[-1]}"
    #       f"\nLoyds mw vor Time\u00B2: {loyds_mw_voronoi_time[-1]}"
    #       f"\nLoyds equal speed Time\u00B2: {loyds_time_speed_eq[-1]}"
    #       )
