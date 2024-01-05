from typing import Tuple
import cv2 as cv
import numpy as np
from numpy import sqrt
import geometric_utility
from geometry import Ellipse


# https://math.stackexchange.com/a/4636320/1125695
def point_ellipse_distance(ellipse: Ellipse, p: Tuple[float, float]) -> float:
    cx = ellipse.center.x
    cy = ellipse.center.y
    rx = ellipse.x_size/2
    ry = ellipse.y_size/2
    # ellipse[2] degrees from rotated to straight = anti-clockwise

    # ellipse center became (0,0)
    px = p[0] - cx
    py = p[1] - cy

    # Rotate point
    radius, angle = geometric_utility.cartesian_to_polar(px, py)
    px, py = geometric_utility.polar_to_cartesian(radius, angle + ellipse.angle)

    px2 = pow(px, 2)
    py2 = pow(py, 2)
    rx2 = pow(rx, 2)
    ry2 = pow(ry, 2)
    dist_center = sqrt(px2 + py2)
    num = rx2 * ry2 * dist_center
    den = px2 * ry2 + py2 * rx2

    return abs(dist_center - sqrt(num/den))
