import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as colors

import csv
import json

import matplotlib.patches as patches
import matplotlib.transforms as transforms

# to make the gif:
import imageio

images = []

# # other way of making the gif
# from PIL import Image

import os
import datetime

number_of_tests = 15

for test in range(number_of_tests):
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time in the desired way
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the file name with the desired format
    dir_files = "run_" + formatted_date

    os.makedirs(dir_files, exist_ok=True)

    from classes import Plane, Robots
    from voronoi_mw import VoronoiMW
    from export_summary_PNG import create_combined_image

    plane = Plane(
        [0, 10],
        [0, 10]
    )

    # Create the mesh grid
    resolution_x = ((plane.x_max - plane.x_min) * 20 + 1)
    resolution_y = ((plane.y_max - plane.y_min) * 20 + 1)
    x, y = np.meshgrid(
        np.linspace(*plane.return_x_limits(), resolution_x),
        np.linspace(*plane.return_y_limits(), resolution_y),
    )


    # def gaussian_2d(x, y, x0, y0, xsig, ysig):
    #     return np.exp(-10 * (((x - x0) / xsig) ** 2 + ((y - y0) / ysig) ** 2))


    def gaussian_2d(x, y, x0, y0, xsig, ysig):
        xsig_top = 0.5 * xsig
        ysig_top = 1.5 * ysig
        x0 *= 0.5
        y0 *= 1.5
        x0_bottom = x0 * 1.5
        y0_bottom = y0 * 0.5
        xsig_bottom = 1.5 * xsig
        ysig_bottom = 0.5 * ysig

        top_quadrant = np.exp(-8 * (((x - x0) / xsig_top) ** 2 + ((y - y0) / ysig_top) ** 2))
        bottom_quadrant = np.exp(-8 * (((x - x0_bottom) / xsig_bottom) ** 2 + ((y - y0_bottom) / ysig_bottom) ** 2))

        return top_quadrant + bottom_quadrant


    x_center = (plane.x_max + plane.x_min) / 2
    y_center = (plane.y_max + plane.y_min) / 2
    x_sigma = (plane.x_max - plane.x_min) / 2
    y_sigma = (plane.y_max - plane.y_min) / 2

    z = gaussian_2d(x, y, x_center, y_center, x_sigma, y_sigma)

    # normalize z so the total is equal to 1
    z /= z.sum()

    # choose random or static for testing:
    random = True
    always_show = False

    # example of some positions the robot could have, this could be randomised
    if random:
        number_of_random_points = 5
        x_random = np.random.uniform(plane.x_min, plane.x_max, number_of_random_points)
        y_random = np.random.uniform(plane.y_min, plane.y_max, number_of_random_points)
        positions = np.column_stack((x_random, y_random))
        # speed_robots = np.random.randint(1, 4, number_of_random_points)
        speed_robots = [
            3,
            3,
            2,
            2,
            1
        ]
    else:  # static
        positions = np.array([
            [2.5, 1.5],
            [1, 8],
            [8, 8],
            [8, 1]
        ])
        speed_robots = np.array([
            1,
            3,
            1,
            3,
        ])

    robots = Robots(plane, positions, speed_robots)

    print(
        np.array(["x", "y", "v"]),
        "\n",
        robots.robot_p_and_v_array()
    )

    # makes sure there are the same amount of random colours as there are robots
    all_colors = list(colors.CSS4_COLORS.values())
    np.random.shuffle(all_colors)
    colors_robots = all_colors[:robots.number_of_robots()]

    all_black = True

    if all_black:
        del colors_robots
        colors_robots = []
        for i in range(robots.number_of_robots()):
            colors_robots.append("black")


    # # creates (for now empty) voronois for every robot voronois = [VoronoiMW(robots.return_position(i),
    # robots.return_max_speed(i)) for i in range(robots.number_of_robots())]


    def assign_robot2voronoi(avg_response_time_i=0):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                grid_coordinate = np.array([x[i, j], y[i, j]])
                time = 999999
                fastest_robot = 0
                for k in range(robots.number_of_robots()):
                    time_robot = np.linalg.norm(robots.return_position(k) - grid_coordinate) / robots.return_max_speed(k)
                    if time_robot < time:
                        fastest_robot = k
                        time = time_robot
                        avg_response_time_i += time_robot * z[i, j]

                voronois[fastest_robot].add_grid_point(grid_coordinate, float(z[i, j]), i, j)

        return avg_response_time_i


    def get_border_voronoi():
        # start with making a grid the exact same size as z, but filled with zeros
        total_grid = np.zeros_like(z)

        # iterate for every voronoi
        for i, voronoi in enumerate(voronois):
            array_i = np.copy(voronoi.grid_coordinates)

            # for every index in vor make it equal to the number of which voronoi it is,
            # so the first plane of voronoi 1 gets filled with only 1's, second one with only 2's, etc
            for j in range(len(array_i)):
                total_grid[voronoi.index_x[j], voronoi.index_y[j]] = i + 1

        # Because every voronoi has a different number for the plane there excist a gradient between those planes,
        # calculate what the gradient is and the if any gradient exist make it equal to 1, otherwise make 'almost' 0 = 0
        grad = np.gradient(total_grid)
        grad = np.where(np.isclose(grad, 0, atol=1e-8), 0, 1)

        # adds the gradient in y and in x direction together so a border forms when plotted
        border = grad[0] + grad[1]

        # makes sure all values are ether 0 or 1 so the border can be plotted with a homogenous colour
        border = np.where(np.isclose(border, 0, atol=1e-8), 0, 1)

        # print(border)
        return border


    def plot_current(iteration, last_iteration=False):
        plt.clf()  # clears the previous picture, so not all 15 appear at once
        plt.figure()

        # plot the robot positions
        plt.scatter(*zip(*robots.positions), c=colors_robots)

        d_letter = 0.22
        for i in range(len(robots.robot_positions())):
            # plot the robot speed for every robot
            plt.text(robots.return_position(i)[0] + d_letter, robots.return_position(i)[1] - d_letter,
                     f"${int(robots.return_max_speed(i))}$",
                     fontsize=9)  # maybe in the text: v{i + 1} =

        # plot the gradient of the pdf (this is z)
        plt.imshow(z, origin='lower', extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max), alpha=0.5)
        plt.colorbar()

        # plot the center point of mass
        for voronoi in voronois:
            plt.plot(*voronoi.center_of_mass(), 'x', c="blue")

        # plot the arrow of the gradient descent, with arrow_scale as variable
        for voronoi in voronois:
            px, py = arrow_scale * voronoi.gradient_descent()
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
        plt.imshow(get_border_voronoi(), cmap=cmap, norm=norm, extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max),
                   origin='lower')

        # saves the iteration in the folder, and append to images for the gif
        # save the plot as png
        filename = f'{dir_files}/{iteration}.png'

        # save the png with the title
        plt.title(f'{iteration}')
        plt.savefig(filename, dpi=150)  # dpi is the resolution of each png

        # read the png image file and append it to the list, without title
        images.append(imageio.v3.imread(filename))

        # show the total plot
        if (iteration % 15) == 0:
            plt.show()
        elif always_show:
            plt.show()
        elif last_iteration:
            plt.show()


    # ---------------------------------------------------------------------------------------------------------------------
    # These are the most important variables to set!
    dt = 0.4
    iterations = 1000
    stop_criterion = 0.009
    p_dot_max = None
    arrow_scale = 7
    # ---------------------------------------------------------------------------------------------------------------------
    avg_response_time = []
    p_dot_list = []
    quickest_response_time = 999999

    for i in range(iterations):
        print(f"\nCalculating iteration {i} \nCurrent p_dot_max: {p_dot_max}, stopping if p_dot_max < {stop_criterion}")

        # creates (for now empty) voronois for every robot
        voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                    range(robots.number_of_robots())]

        avg_response_time.append(assign_robot2voronoi())

        # this plot is with the current position and current location
        plot_current(i)

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
                f"\nThe max p dot ({p_dot_max}) if smaller than {stop_criterion}, iteration stopped. \nStopped at iteration: {i}")

            # makes sure the final plot get shown
            voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                        range(robots.number_of_robots())]

            assign_robot2voronoi()
            plot_current(i, last_iteration=True)
            break

        # deletes the voronois so it can be used again next iteration
        del voronois


    indices = list(range(len(avg_response_time)))

    plt.clf()
    plt.figure()
    plt.plot(indices, avg_response_time, marker='o')

    plt.title('Plot of average response time')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')

    plt.grid(True)

    plt.savefig(f"{dir_files}/Avg_response_time.png", dpi=150)
    plt.show()

    # show the velocity vector of all the robots over time
    plt.clf()
    plt.figure()
    p_dot_robots = list(map(list, zip(*p_dot_list)))
    for i, p_dots in enumerate(p_dot_robots):
        plt.plot(p_dots, label=f"Robot {i + 1}")

    plt.axhline(y=stop_criterion, color='r', linestyle='-', label='Stop criterion')

    plt.title('Plot of velocity vector of all robots')
    plt.xlabel('Iteration')
    plt.ylabel('Pdot (m/s)')

    plt.legend()
    plt.savefig(f"{dir_files}/Velocity_robots.png", dpi=150)
    plt.show()

    # imageio.mimwrite(f'{dir_files}/output2.mp4', images, fps=2)
    index_min = np.argmin(np.array(avg_response_time))
    print(f"The average response time was the quickest at index: {index_min}. (The last index is: {len(avg_response_time) - 1})"
          f"\nThe time at this index was: {avg_response_time[index_min]}. The time at the last index was: {avg_response_time[-1]}"
          f"\n"
          f"\nThe average response time at the start was {avg_response_time[0]}."
          )

    # create the gif:
    imageio.mimsave(f'{dir_files}/gif_speed1.gif', images, duration=0.9)
    imageio.mimsave(f'{dir_files}/gif_speed2.gif', images, duration=0.5)
    imageio.mimsave(f'{dir_files}/gif_speed3.gif', images, duration=0.3)
    imageio.mimsave(f'{dir_files}/gif_speed4.gif', images, duration=0.18)

    # save avg_response_time and p_dot_list to csv
    with open(f'{dir_files}/avg_response_time.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(avg_response_time)

    with open(f'{dir_files}/p_dot_list.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(p_dot_list)

    index_fastes_time = avg_response_time.index(min(avg_response_time))

    # save start time, end time, quickest time and index of fastest time
    times_data = {
        'start_time': avg_response_time[0],
        'end_time': avg_response_time[-1],
        'quickest_time': min(avg_response_time),
        'index_fastest_time': index_fastes_time
    }

    with open(f'{dir_files}/times.json', 'w') as file:
        json.dump(times_data, file, indent=4)

    # save the location of the robots with their starting position
    robots_data = {
        f"Robot {i + 1}": {
            "x": robots.return_position(i)[0],
            "y": robots.return_position(i)[1],
            "v": robots.return_max_speed(i)
        }
        for i in range(robots.number_of_robots())
    }

    with open(f'{dir_files}/robots.json', 'w') as file:
        json.dump(robots_data, file, indent=4)

    # create an overview of the most important data. So a test can be analysed quickly
    create_combined_image(
        start_index_png=f'{dir_files}/{0}.png',
        fastest_index_png=f'{dir_files}/{f"{index_fastes_time}"}.png',
        end_index_png=f'{dir_files}/{len(avg_response_time) - 1}.png',
        average_response_time_png=f"{dir_files}/Avg_response_time.png",
        velocity_robots_png=f"{dir_files}/Velocity_robots.png",
        start_time=avg_response_time[0],
        fastest_time=min(avg_response_time),
        end_time=avg_response_time[-1],
        robot_info_list=robots.robot_p_and_v_array(),
        output_path=f"{dir_files}/data_overview.png"
    )

    # rename the plot with the best index, so it can be found quickly
    os.rename(
        f'{dir_files}/{f"{index_fastes_time}"}.png',
        f'{dir_files}/{f"{index_fastes_time} - lowest average response time"}.png'
    )
