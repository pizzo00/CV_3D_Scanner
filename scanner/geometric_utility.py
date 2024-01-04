from typing import Tuple, List
import numpy as np


def magnitude(x: float, y: float, z: float = 0.0) -> float:
    return np.sqrt(x * x + y * y + z * z)


# https://stackoverflow.com/a/47823499/7490904
def get_line_intersection(line1_point1: List[float], line1_point2: List[float], line2_point1: List[float], line2_point2: List[float]):
    A = np.array([line1_point1, line1_point2])
    B = np.array([line2_point1, line2_point2])
    t, s = np.linalg.solve(np.array([A[1] - A[0], B[0] - B[1]]).T, B[0] - A[0])
    return (1 - t) * A[0] + t * A[1]


def get_plane(point1, point2, point3):
    v1 = point1 - point2
    v2 = point2 - point3
    cross = np.cross(v1, v2)
    cross = cross / magnitude(cross[0], cross[1], cross[2])
    k = -np.sum(point1*cross)
    return np.array([cross[0], cross[1], cross[2], k])


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
