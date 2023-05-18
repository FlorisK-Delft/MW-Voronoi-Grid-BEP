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

    def gradient_descent(self):
        return 4*(-2/(self.speed**2)) * self.total_mass() * (self.robot - self.center_of_mass())

    def position(self):
        return self.robot[0], self.robot[1]
