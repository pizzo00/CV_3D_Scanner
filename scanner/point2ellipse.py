# def __get_root(r0: float, z0: float, z1: float, g: float) -> float:
#     n0 = r0*z0
#     s0 = z1-1
#     s1 = 0 if g < 0 else
#     s = 0
#     for
from typing import Tuple

from numpy import sqrt

import polar_utility


def pointEllipseDistance(ellipse: Tuple[Tuple[float, float], Tuple[float, float], float], p: Tuple[float, float]) -> float:
    cx = ellipse[0][0]
    cy = ellipse[0][1]
    rx = ellipse[1][0]
    ry = ellipse[1][1]

    # ellipse center is (0,0)
    px = p[0] - cx
    py = p[1] - cy

    radius, angle = polar_utility.cartesian_to_polar(px, py)
    px, py = polar_utility.polar_to_cartesian(radius - ellipse[2], angle)

    px2 = pow(px, 2)
    py2 = pow(py, 2)
    rx2 = pow(rx, 2)
    ry2 = pow(ry, 2)
    dist_center = sqrt(px2 + py2)
    num = rx2 * ry2 * dist_center
    den = px2 * ry2 + py2 * rx2

    return abs(dist_center - sqrt(num/den))
