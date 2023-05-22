import numpy as np


class VoronoiMW:
    def __init__(self, position_r: np.ndarray, speed: float):
        self.robot = position_r
        self.speed = speed
        self.grid_coordinates = np.empty((0, 3))
        self.index_x = []
        self.index_y = []

    def add_grid_point(self, point: np.ndarray, z: float, index_x, index_y):
        self.index_x.append(index_x)
        self.index_y.append(index_y)
        weighted_coordinate = np.array([float(point[0]), float(point[1]), float(z)])
        weighted_coordinate = weighted_coordinate.reshape(1, -1)
        self.grid_coordinates = np.append(self.grid_coordinates, weighted_coordinate, axis=0)

    def total_mass(self):
        return np.sum(self.grid_coordinates[:, 2])

    def center_of_mass(self):
        total_mass = self.total_mass()
        x_center = np.sum(self.grid_coordinates[:, 0] * self.grid_coordinates[:, 2]) / total_mass
        y_center = np.sum(self.grid_coordinates[:, 1] * self.grid_coordinates[:, 2]) / total_mass
        return x_center, y_center

    def gradient_descent(self, gain=1, loyds=False):
        if loyds:
            return gain * -2 * self.total_mass() * (self.robot - self.center_of_mass())
        else:
            return gain * (-2 / (self.speed ** 2)) * self.total_mass() * (self.robot - self.center_of_mass())

    def position(self):
        return self.robot[0], self.robot[1]


def assign_robot2voronoi(x_mesh, y_mesh, z_mesh, robots_for_vor, init_voronois,
                         loyds=False):
    avg_response_time_i_speed_eq = 0
    avg_response_time_i = 0
    for index_i in range(x_mesh.shape[0]):
        for index_j in range(x_mesh.shape[1]):
            grid_coordinate = np.array([x_mesh[index_i, index_j], y_mesh[index_i, index_j]])
            time = 999999
            fastest_robot = 0
            for k in range(robots_for_vor.number_of_robots()):
                if loyds:
                    time_robot = np.linalg.norm(robots_for_vor.return_position(k) -
                                                grid_coordinate) #  / 1
                    actual_time_robot = np.linalg.norm(robots_for_vor.return_position(k) -
                                   grid_coordinate) / robots_for_vor.return_max_speed(k)
                else:
                    time_robot = np.linalg.norm(robots_for_vor.return_position(k) -
                                                grid_coordinate) / robots_for_vor.return_max_speed(k)
                if time_robot < time:
                    fastest_robot = k
                    time = time_robot
                    if loyds:
                        actual_time_robot_loyds = actual_time_robot

            if loyds:
                avg_response_time_i_speed_eq += time ** 2 * z_mesh[index_i, index_j]
                avg_response_time_i += (actual_time_robot_loyds) ** 2 * z_mesh[index_i, index_j]
            else:
                avg_response_time_i += time ** 2 * z_mesh[index_i, index_j]

            init_voronois[fastest_robot].add_grid_point(grid_coordinate,
                                                        float(z_mesh[index_i, index_j]), index_i, index_j)

    return avg_response_time_i, avg_response_time_i_speed_eq


def response_time_mw_voronoi(x_mesh, y_mesh, z_mesh, robots_class):
    avg_response_time_mw = 0

    for index_i in range(x_mesh.shape[0]):
        for index_j in range(x_mesh.shape[1]):
            grid_coordinate = np.array([x_mesh[index_i, index_j], y_mesh[index_i, index_j]])
            time = 999999
            for k in range(robots_class.number_of_robots()):
                time_robot = np.linalg.norm(robots_class.return_position(k) -
                                            grid_coordinate) / robots_class.return_max_speed(k)
                if time_robot < time:
                    time = time_robot

            avg_response_time_mw += time ** 2 * z_mesh[index_i, index_j]

    return avg_response_time_mw


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
