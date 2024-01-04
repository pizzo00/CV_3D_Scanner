import numpy as np
import cv2 as cv


class Pose:
    @staticmethod
    def invert_pose(r, t):
        return np.matrix(r).T, (-np.matrix(r).T) @ np.matrix(t)

    @staticmethod
    def get_pose_matrix(r, t):
        return np.matrix(
            [
                [r.item(0, 0), r.item(0, 1), r.item(0, 2), t.item(0)],
                [r.item(1, 0), r.item(1, 1), r.item(1, 2), t.item(1)],
                [r.item(2, 0), r.item(2, 1), r.item(2, 2), t.item(2)],
                [0, 0, 0, 1],
            ])

    def __init__(self, r, t):
        self.r = cv.Rodrigues(r)[0]
        self.t = t
        self.ri, self.ti = Pose.invert_pose(self.r, self.t)
        self.m = Pose.get_pose_matrix(self.r, self.t)
        self.mi = Pose.get_pose_matrix(self.ri, self.ti)
