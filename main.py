import numpy as np
import matplotlib.pyplot as plt

from classes import Plane, Robots
from voronoi_mw import VoronoiMW

plane = Plane(
    [0, 10],
    [0, 10]
)

# Create the mesh grid
resolution_x = ((plane.x_max - plane.x_min)*10 + 1)
resolution_y = ((plane.y_max - plane.y_min)*10 + 1)
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


# choose random or static for testing:
random = True

# example of some positions the robot could have, this could be randomised
if random:
    number_of_random_points = 100
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

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        grid_coordinate = np.array([x[i,j], y[i,j]])
        time = 999999
        fastest_robot = 0
        for k in range(robots.number_of_robots()):
            time_robot = np.linalg.norm(robots.return_position(k) - grid_coordinate) / robots.return_max_speed(k)
            if time_robot < time:
                fastest_robot = k
                time = time_robot

        voronois[fastest_robot].add_grid_point(grid_coordinate, float(z[i, j]))

for i in range(len(voronois)):
    print(f"\nCenter point of robot {i}:")
    print(voronois[i].center_of_mass())
    print(voronois[i].total_mass())
    # print(voronois[i].gradient_descent())


