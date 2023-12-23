from enum import Enum
import numpy as np

from polar_utility import polar_to_cartesian


class MarkerColors(Enum):
    Cyan = 'C'
    Yellow = 'Y'
    Magenta = 'M'
    White = 'W'
    Black = 'B'


class CircularMarker:
    _instance = None

    colors = ['Y', 'W', 'M', 'B', 'M', 'M', 'C', 'C', 'C', 'Y', 'W', 'B', 'M', 'Y', 'W', 'B', 'Y', 'W', 'B', 'C']
    _number_of_markers = len(colors)  # = 20
    _radius = 7.5
    _angle_diff = np.deg2rad(360 / 20)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CircularMarker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.points = np.zeros((self._number_of_markers, 3), np.float32)
        for i in range(self._number_of_markers):
            angle = i * self._angle_diff
            x, y = polar_to_cartesian(angle, self._radius)
            self.points[i, 0] = x
            self.points[i, 1] = y
            # self.points[i, 2] = 0

    def get_markers_position(self, search_colors: list[MarkerColors]) -> int | None:
        for i in range(self._number_of_markers):
            if search_colors[0].value == self.colors[i] and \
               search_colors[1].value == self.colors[(i+1) % self._number_of_markers] and \
               search_colors[2].value == self.colors[(i+2) % self._number_of_markers] and \
               search_colors[3].value == self.colors[(i+3) % self._number_of_markers]:
                return i
        return None
