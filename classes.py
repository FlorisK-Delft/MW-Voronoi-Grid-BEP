# The class Plane is used to make a plane and not make it such
# that it can't be added by accident.
import numpy as np
from voronoi_mw import VoronoiMW

class Plane:
    def __init__(self, x_axis: np.ndarray, y_axis: np.ndarray):  # pass in the form: [x min, x max], [y min, y max]
        # assert isinstance(
        #     x_axis, np.ndarray
        # ) and (
        #     p_i.shape == (2,)
        # ) and (
        #     np.issubdtype(p_i.dtype, np.number)
        # ), "x_axis must be a numpy array of shape (2,) and numeric dtype"
        self.x_min = x_axis[0]
        self.x_max = x_axis[1]
        self.y_min = y_axis[0]
        self.y_max = y_axis[1]

    def return_limits(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    def return_x_limits(self):
        return self.x_min, self.x_max

    def return_y_limits(self):
        return self.y_min, self.y_max


def position_within_operating_plane_speed_non_zero(position, operating_plane, speed):
    if (  # look if it is within the x-axis border (and important, not ON the border!)
            operating_plane.x_min < position[0] < operating_plane.x_max
    ) and (  # look if it is within the y-axis border (and important, not ON the border!)
            operating_plane.y_min < position[1] < operating_plane.y_max
    ) and (
            speed > 0
    ):
        return True
    else:  # if not
        return False


class Robots:
    def __init__(self, plane, start_positions, speed_robots):
        self.operating_plane = plane

        # start with an empty matrix in the correct 0x2 form:
        self.positions = np.array([]).reshape(0, 2)
        self.speed_robots = np.array([])

        for i in range(len(start_positions)):
            if position_within_operating_plane_speed_non_zero(start_positions[i], self.operating_plane,
                                                              speed_robots[i]):
                self.positions = np.append(self.positions, [start_positions[i]], axis=0)
                self.speed_robots = np.append(self.speed_robots, speed_robots[i])

            else:
                print(f"Position ({start_positions[i][0]}, {start_positions[i][0]}) does not fit within the plane.")

    def add_robot(self, position, speed):
        # pass a 2x1 numpy array with x and y coordinate
        if position_within_operating_plane_speed_non_zero(position, self.operating_plane, speed):
            self.positions = np.append(self.positions, [position], axis=0)
            self.speed_robots = np.append(self.speed_robots, speed_robots[i])
        else:
            print(f"Position ({position[0]}, {position[1]}) does not fit within the plane.")

    def add_multiple_robots(self, multiple_positions, multiple_speeds):
        # pass a list of 2x1 numpy arrays with x and y coordinates
        for i in range(len(multiple_positions)):
            if position_within_operating_plane_speed_non_zero(multiple_positions[i], self.operating_plane, multiple_speeds[i]):
                self.positions = np.append(self.positions, [multiple_positions[i]], axis=0)
                self.speed_robots = np.append(self.speed_robots, multiple_speeds[i])
            else:
                print(
                    f"Position ({multiple_positions[i][0]}, {multiple_positions[i][1]}) does not fit within the plane.")

    def robot_positions(self):
        # returns the positions as a (2d) numpy array.
        return self.positions

    def robot_speeds(self):
        return self.speed_robots

    def robot_p_and_v_array(self):
        return np.column_stack((self.positions, self.speed_robots))

    def number_of_robots(self):
        return len(self.positions)

    def return_position(self, index: int):
        return self.positions[index]

    def return_max_speed(self, index: int):
        return self.speed_robots[index]

    def update_positions(self, position, index):
        self.positions[index] = position

    def time_step(self, p_dot, dt, index):
         new_position = self.positions[index] + p_dot * dt
         self.positions[index] = new_position

    def time_step_all(self, voronois, dt):
        p_dot_list_i = []
        p_dot_max = 0

        for i in range(len(self.positions)):
            p_dot = voronois[i].gradient_descent()
            new_position = self.positions[i] + p_dot * dt
            self.positions[i] = new_position
            if np.linalg.norm(p_dot) > p_dot_max:
                p_dot_max = np.linalg.norm(p_dot)
            p_dot_list_i.append(np.linalg.norm(p_dot))

        return p_dot_list_i, p_dot_max


    # def sort_positions_xy(self): ! does not work, then the speeds should be sorted accordingly.
    #     # sort the positions first by x and then by y
    #     sorted_indices = np.lexsort((self.positions[:, 1], self.positions[:, 0]))
    #     self.positions = self.positions[sorted_indices]
