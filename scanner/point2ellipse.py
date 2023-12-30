# def __get_root(r0: float, z0: float, z1: float, g: float) -> float:
#     n0 = r0*z0
#     s0 = z1-1
#     s1 = 0 if g < 0 else
#     s = 0
#     for
from typing import Tuple
import cv2 as cv
import numpy as np
from numpy import sqrt
import polar_utility
from geometry import Ellipse


def point_ellipse_distance(ellipse: Ellipse, p: Tuple[float, float]) -> float:
    cx = ellipse.center.x
    cy = ellipse.center.y
    rx = ellipse.x_size/2
    ry = ellipse.y_size/2
    # ellipse[2] from rotated to straight = anti-clockwise

    # ellipse center is (0,0)
    px = p[0] - cx
    py = p[1] - cy

    radius, angle = polar_utility.cartesian_to_polar(px, py)
    px, py = polar_utility.polar_to_cartesian(radius, angle + ellipse.angle)

    px2 = pow(px, 2)
    py2 = pow(py, 2)
    rx2 = pow(rx, 2)
    ry2 = pow(ry, 2)
    dist_center = sqrt(px2 + py2)
    num = rx2 * ry2 * dist_center
    den = px2 * ry2 + py2 * rx2

    return abs(dist_center - sqrt(num/den))
