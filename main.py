import numpy as np
import matplotlib.pyplot as plt

from classes import Plane, Robots
from voronoi_mw import VoronoiMW

plane = Plane(
    [0, 10],
    [0, 10]
)

# Create the mesh grid
resolution_x = ((plane.x_max - plane.x_min) * 10 + 1)
resolution_y = ((plane.y_max - plane.y_min) * 10 + 1)
x, y = np.meshgrid(
    np.linspace(*plane.return_x_limits(), resolution_x),
    np.linspace(*plane.return_y_limits(), resolution_y),
)


def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5 * (((x - x0) / xsig) ** 2 + ((y - y0) / ysig) ** 2))


x_center = (plane.x_max + plane.x_min) / 2
y_center = (plane.y_max + plane.y_min) / 2
x_sigma = (plane.x_max - plane.x_min) / 2
y_sigma = (plane.y_max - plane.y_min) / 2

z = gaussian_2d(x, y, x_center, y_center, x_sigma, y_sigma)

# normalize z so the total is equal to 1
z /= z.sum()

# choose random or static for testing:
random = True

# example of some positions the robot could have, this could be randomised
if random:
    number_of_random_points = 5
    x_random = np.random.uniform(plane.x_min, plane.x_max, number_of_random_points)
    y_random = np.random.uniform(plane.y_min, plane.y_max, number_of_random_points)
    positions = np.column_stack((x_random, y_random))
    speed_robots = np.random.randint(1, 4, number_of_random_points)
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


# # creates (for now empty) voronois for every robot
# voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in range(robots.number_of_robots())]


def assign_robot2voronoi():
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

            voronois[fastest_robot].add_grid_point(grid_coordinate, float(z[i, j]), i, j)


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

    print(border)
    return border


def plot_current(iteration):
    plt.figure()
    plt.scatter(*zip(*robots.positions), c=colors_robots)

    # plot the gradient of the pdf (this is z)
    plt.imshow(z, origin='lower', extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max), alpha=0.5)
    plt.colorbar()

    # plot the voronoi boundary's
    cmap = colors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(get_border_voronoi(), cmap=cmap, norm=norm, extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max),
               origin='lower')

    # saves the iteration in the folder
    plt.savefig(f'{this_test}/{f"{iteration + 1}"}.png')

    # show the total plot
    plt.show()


dt = 0.3
iterations = 1000
stop_criterion = 0.1

for i in range(iterations):
    print(f"\nCalculating iteration {i}")

    # creates (for now empty) voronois for every robot
    voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                range(robots.number_of_robots())]

    assign_robot2voronoi()

    # this plot is with the current position and current location
    plot_current(i)

    # calculate the next positions, return the maximum robot displacement p_dot to look for stop criterion
    p_dot_max = robots.time_step_all(voronois, dt)

    # stop criterion, if the vector for moving the robots gets small enough, stop moving the robots.
    if p_dot_max < stop_criterion:
        print(f"\nThe max p dot ({p_dot_max}) if smaller than 0.01, iteration stopped. \nStopped at iteration: {i}")

        # makes sure the final plot get shown
        voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                    range(robots.number_of_robots())]

        assign_robot2voronoi()
        plot_current(i)
        break

    # deletes the voronois so it can be used again next iteration
    del voronois
