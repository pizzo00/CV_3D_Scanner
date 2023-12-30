import cv2 as cv
import numpy as np


class Point:
    def __init__(self, x: float, y: float):
        self.raw = np.array([x, y])
        self.x = x
        self.y = y


class Ellipse:
    def __init__(self, ellipse: cv.typing.RotatedRect):
        self.raw = ellipse

        self.center = Point(ellipse[0][0], ellipse[0][1])

        self.x_size = ellipse[1][0]
        self.y_size = ellipse[1][1]
        self.max_size = max(self.x_size, self.y_size)
        self.min_size = min(self.x_size, self.y_size)

        self.angleDeg = ellipse[2]
        self.angle = np.deg2rad(self.angleDeg)


class Rectangle:
    def __init__(self, rect: cv.typing.Rect, points: cv.typing.MatLike):
        self.raw = rect
        self.points = points

        self.center = Point(rect[0], rect[1])
        self.x_size = rect[2]
        self.y_size = rect[3]
        self.area = self.x_size * self.y_size
