from enum import Enum
from typing import List, Any, Tuple

import numpy as np

from polar_utility import polar_to_cartesian


class MarkerColors(Enum):
    Cyan = 'C'
    Yellow = 'Y'
    Magenta = 'M'
    White = 'W'
    Black = 'B'

    @staticmethod
    def get_from_pixel_debug(pixel: List[float]) -> Tuple[int, int, int]:
        threshold = 75
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]

        #return (int(r), int(g), int(b))

        rt = r > threshold
        gt = g > threshold
        bt = b > threshold

        if rt and gt and bt:
            return (255, 255, 255)
        if gt and bt:
            return (0, 255, 255)
        if rt and gt:
            return (255, 255, 0)
        if rt and bt:
            return (255, 0, 255)
        return (0, 0, 0)
        # TODO return none

    @staticmethod
    def get_from_pixel(pixel: List[float]) -> 'MarkerColors':
        threshold = 75
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]
        rt = r > threshold
        gt = g > threshold
        bt = b > threshold

        if rt and gt and bt:
            return MarkerColors.White
        if gt and bt:
            return MarkerColors.Cyan
        if rt and gt:
            return MarkerColors.Yellow
        if rt and bt:
            return MarkerColors.Magenta
        return MarkerColors.Black
        # TODO return none


class CircularMarker:
    _instance = None

    colors = ['Y', 'W', 'M', 'B', 'M', 'M', 'C', 'C', 'C', 'Y', 'W', 'B', 'M', 'Y', 'W', 'B', 'Y', 'W', 'B', 'C']
    _number_of_markers = len(colors)  # = 20
    _radius = 750
    _angle_diff = np.deg2rad(360 / 20)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CircularMarker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.points = np.zeros((self._number_of_markers, 3), np.float32)
        for i in range(self._number_of_markers):
            angle = i * self._angle_diff
            x, y = polar_to_cartesian(self._radius, angle)
            self.points[i, 0] = x
            self.points[i, 1] = y
            # self.points[i, 2] = 0

    def get_marker_point(self, idx: int):
        return self.points[idx % self._number_of_markers]

    def get_marker_color(self, idx: int):
        return self.colors[idx % self._number_of_markers]

    def get_markers_point(self, idx: int):
        return self.points[idx % self._number_of_markers]

    def get_markers_points(self, idx: int):
        return [
            self.points[idx % self._number_of_markers],
            self.points[(idx+1) % self._number_of_markers],
            self.points[(idx+2) % self._number_of_markers],
            self.points[(idx+3) % self._number_of_markers]
        ]

    def get_markers_position(self, search_colors: list[MarkerColors]) -> int | None:
        for i in range(self._number_of_markers):
            if search_colors[0].value == self.colors[i % self._number_of_markers] and \
               search_colors[1].value == self.colors[(i+1) % self._number_of_markers] and \
               search_colors[2].value == self.colors[(i+2) % self._number_of_markers] and \
               search_colors[3].value == self.colors[(i+3) % self._number_of_markers]:
                return i
        return None
