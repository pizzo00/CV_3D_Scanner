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
