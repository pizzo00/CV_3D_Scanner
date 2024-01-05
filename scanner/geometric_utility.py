from typing import Tuple, List
import numpy as np


def magnitude(x: float, y: float, z: float = 0.0) -> float:
    return np.sqrt(x * x + y * y + z * z)


# https://stackoverflow.com/a/47823499/7490904
def get_lines_intersection(a0: List[float], a1: List[float], b0: List[float], b1: List[float]):
    a = np.array([a0, a1])
    b = np.array([b0, b1])
    t, _ = np.linalg.solve(np.array([a[1] - a[0], b[0] - b[1]]).T, b[0] - a[0])
    return (1 - t) * a[0] + t * a[1]


# https://stackoverflow.com/a/18543221/7490904
def get_line_plane_intersection(l0: np.ndarray[float], l1: np.ndarray[float], plane, epsilon=1e-6):
    u = l1 - l0
    plane_xyz = np.array([plane[0], plane[1], plane[2]])
    dot = np.dot(plane_xyz, u)
    if abs(dot) > epsilon:
        p_co = plane_xyz * (-plane[3] / np.dot(plane_xyz, plane_xyz))

        w = l0 - p_co
        fac = -np.dot(plane_xyz, w) / dot
        return l0 + (u * fac)

    return None


# https://math.stackexchange.com/a/2686620/1125695
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


def get_angle(center: Tuple[float, float], point: Tuple[float, float]) -> float:
    x = point[0] - center[0]
    y = point[1] - center[1]
    _, angle = cartesian_to_polar(x, y)
    return angle


# https://community.esri.com/t5/python-blog/point-in-polygon-geometry-mysteries/ba-p/893890
def is_inside_polygon(pnts, poly, return_winding=False):
    """Return points in polygon using a winding number algorithm in numpy.

    Parameters
    ----------
    pnts : Nx2 array
        Points represented as an x,y array.
    poly : Nx2 array
        Polygon consisting of at least 4 points oriented in a clockwise manner.
    return_winding : boolean
        True, returns the winding number pattern for testing purposes.  Keep as
        False to avoid downstream errors.

    Returns
    -------
    The points within or on the boundary of the geometry.

    References
    ----------
    `<https://github.com/congma/polygon-inclusion/blob/master/
    polygon_inclusion.py>`_.  inspiration for this numpy version
    """
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x, y = pnts.T         # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = (y_y0 >= 0.0)
    chk2 = np.less(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
    chk3 = np.sign(diff_).astype('int')
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    out_ = pnts[np.nonzero(wn)]
    if return_winding:
        return out_, wn
    return out_