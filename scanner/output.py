import time

import numpy as np


class OutputXYZ:
    def __init__(self):
        self.path = '.\\data\\output\\' + time.strftime("%Y-%m-%d_%H.%M.%S") + '.xyz'
        self.stream = open(self.path, "w")

    def add_point(self, point: np.ndarray[float]):
        if point[2] > 5:
            self.stream.write("{:.2f} {:.2f} {:.2f}\n".format(point[0], point[1], point[2]))
