import time
import numpy as np
import open3d as o3d

from parameters import Parameters
from pose import Pose


class OutputXYZ:
    points = []

    def __init__(self, filename: str):
        self.path = filename + '.xyz'
        self.stream = open(self.path, "w")

        if Parameters.win_3d:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(height=600, width=350)

    def add_point(self, point: np.ndarray[float]):
        if point[2] > 5:
            if Parameters.win_3d:
                self.points.append(list(point))
            self.stream.write("{:.2f} {:.2f} {:.2f}\n".format(point[0], point[1], point[2]))

    def print_preview(self, pose: Pose):
        if Parameters.win_3d:
            # Add geometry
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
            self.vis.add_geometry(cloud)

            # Set pose
            ctr = self.vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            params.extrinsic = pose.m
            ctr.convert_from_pinhole_camera_parameters(params, True)

            # Update view
            self.vis.poll_events()
            self.vis.update_renderer()
            self.points = []

