from typing import Tuple

import numpy as np


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    radius = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    return radius, angle


def polar_to_cartesian(radius: float, angle: float) -> Tuple[float, float]:
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y


def get_angle(center: Tuple[float, float], point: Tuple[float, float]):
    x = point[0] - center[0]
    y = point[1] - center[1]
    _, angle = cartesian_to_polar(x, y)
    return angle
