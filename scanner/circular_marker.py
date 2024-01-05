from enum import Enum
from typing import List, Any, Tuple

import numpy as np

import geometric_utility


class MarkerColors(Enum):
    Cyan = 'C'
    Yellow = 'Y'
    Magenta = 'M'
    White = 'W'
    Black = 'B'
    NoneColor = 'N'

    @staticmethod
    def _get_pixel_safe(image, x: int, y: int) -> List[float] | None:
        if image is None:
            return None
        h, w = image.shape[:2]
        if 0 < x < w and 0 < y < h:
            return image[y][x]
        return None

    @staticmethod
    def get_from_pixel_debug(image, x: int, y: int) -> Tuple[int, int, int]:
        color = MarkerColors.get_from_pixel(image, x, y)
        if MarkerColors.White:
            return 255, 255, 255
        if MarkerColors.Cyan:
            return 255, 255, 0
        if MarkerColors.Yellow:
            return 0, 255, 255
        if MarkerColors.Magenta:
            return 255, 0, 255
        if MarkerColors.Black:
            return 0, 0, 0
        return 150, 150, 150

    @staticmethod
    def get_from_pixel(image, x: int, y: int) -> 'MarkerColors':
        pixel = MarkerColors._get_pixel_safe(image, x, y)
        if pixel is None:
            return MarkerColors.NoneColor

        threshold = 75
        r = pixel[2]
        g = pixel[1]
        b = pixel[0]
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
        if not rt and not gt and not bt:
            return MarkerColors.Black
        return MarkerColors.NoneColor


class CircularMarker:
    _instance = None

    colors = ['Y', 'W', 'M', 'B', 'M', 'M', 'C', 'C', 'C', 'Y', 'W', 'B', 'M', 'Y', 'W', 'B', 'Y', 'W', 'B', 'C']
    _number_of_markers = len(colors)  # = 20
    _radius = 75
    _angle_diff = np.deg2rad(360 / 20)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CircularMarker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.points = np.zeros((self._number_of_markers, 3), np.float32)
        for i in range(self._number_of_markers):
            angle = i * self._angle_diff
            x, y = geometric_utility.polar_to_cartesian(self._radius, angle)
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

    @staticmethod
    def get_laser_search_area(camera_x: float, camera_y: float):
        intersection_x = camera_x
        intersection_y = camera_y
        radius, angle = geometric_utility.cartesian_to_polar(intersection_x, intersection_y)

        search_angle = np.deg2rad(60)
        angle1 = angle - search_angle/2
        angle2 = angle + search_angle/2
        radius1 = CircularMarker._radius * 22/40
        radius2 = CircularMarker._radius * 32/40

        p1 = geometric_utility.polar_to_cartesian(radius1, angle1)
        p2 = geometric_utility.polar_to_cartesian(radius1, angle2)
        p3 = geometric_utility.polar_to_cartesian(radius2, angle1)
        p4 = geometric_utility.polar_to_cartesian(radius2, angle2)

        # Add z = 0
        return [[i[0], i[1], 0.0] for i in [p1, p2, p3, p4]]
