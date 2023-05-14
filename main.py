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
    speed_robots = np.random.uniform(1, 2, number_of_random_points)
    # speed_robots = np.array([
    #     1,
    #     1,
    #     1,
    #     3,
    #     3,
    #     2
    # ])
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

voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in range(robots.number_of_robots())]


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


# for i in range(len(voronois)):
#     print(f"\nCenter point of robot {i}:")
#     print(voronois[i].center_of_mass())
#     print(voronois[i].total_mass())
#     print(voronois[i].gradient_descent())


# for i in range(robots.number_of_robots):
#     robots.time_step(voronois[i].gradient_descent(), dt, i)

# robots.time_step_all(voronois, dt)
dt = 0.3
iterations = 1000
plt.figure()

all_colors = list(colors.CSS4_COLORS.values())
np.random.shuffle(all_colors)
colors_robots = all_colors[:robots.number_of_robots()]

for i in range(iterations):
    plt.scatter(*zip(*robots.positions), c=colors_robots)
    del voronois
    voronois = [VoronoiMW(robots.return_position(i), robots.return_max_speed(i)) for i in
                range(robots.number_of_robots())]

    assign_robot2voronoi()

    prev_positions = robots.positions
    p_dot_max = robots.time_step_all(voronois, dt)

    if p_dot_max < 0.2:
        print(f"\nThe max p dot ({p_dot_max}) if smaller than 0.01, iteration stopped. \nStopped at iteration: {i}")
        break

    #
    # if min(np.linalg.norm(prev_positions - robots.positions)) < 0.01:
    #     break
    # break
    # print(f"\nNew position after the {i+1} iteration:")
    # print(robots.positions)

plt.scatter(*zip(*robots.positions), c=colors_robots)

plt.imshow(z, origin='lower', extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max), alpha=0.5)
plt.colorbar()  # optionally add a colorbar to show the intensity scale
# plt.show()

total_grid = np.zeros_like(z)

for i in range(len(voronois)):
    array_i = np.copy(voronois[i].grid_coordinates)

    for j in range(len(array_i)):
        total_grid[voronois[i].index_x[j], voronois[i].index_y[j]] = i + 1

grad = np.gradient(total_grid)
grad = np.where(np.isclose(grad, 0, atol=1e-8), 0, 1)

border = grad[0] + grad[1]

border = np.where(np.isclose(border, 0, atol=1e-8), 0, 1)

cmap = colors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])
bounds = [0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(border, cmap=cmap, norm=norm, extent=(plane.x_min, plane.x_max, plane.y_min, plane.y_max), origin='lower')
plt.show()