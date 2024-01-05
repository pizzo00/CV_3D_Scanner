import random
import time
from typing import List, Tuple

import parameters
from output import OutputXYZ
from parameters import Parameters
import point2ellipse
import geometric_utility
from circular_marker import CircularMarker, MarkerColors
import numpy as np
import cv2 as cv
import imutils

from geometry import Rectangle, Ellipse
from pose import Pose
from undistorion import undistort_image, get_camera_matrix, get_distortion, get_h_w
from wall_marker import WallMarker

# step_by of laser scan (in pixel)
SCANNER_INTERVAL = 5

# step_by when laser lines is fitted detection (in pixel)
LASER_PLANE_FIT_INTERVAL_X = 10
LASER_PLANE_FIT_INTERVAL_Y = 3


def get_info_solvepnp():
    camera_matrix = get_camera_matrix()
    distortion = np.zeros((4, 1))  # get_distortion()
    return camera_matrix, distortion


# *
def back_projection_plane(img_x: float, img_y: float, plane, pose: Pose):
    camera_matrix, distortion = get_info_solvepnp()

    # Get frame point
    point_undistorted = cv.undistortPoints(np.array([img_x, img_y]), camera_matrix, distortion)[0][0]
    x = point_undistorted[0]
    y = point_undistorted[1]
    z = 1  # frame has z = 1
    frame_point = np.matrix([x, y, z, 1]).T

    # Convert frame point in world coordinates
    frame_point = pose.mi @ frame_point
    frame_point = np.array(frame_point[:-1]).flatten() / frame_point.item(3)

    # Get intersection of ray with the plane
    world_point = geometric_utility.get_line_plane_intersection(np.array(pose.ti).flatten(), frame_point, plane)
    return world_point


# *
def back_projection_z(img_x: float, img_y: float, world_z: float, pose: Pose):
    camera_matrix, distortion = get_info_solvepnp()

    # Get frame point
    point_undistorted = cv.undistortPoints(np.array([img_x, img_y]), camera_matrix, distortion)[0][0]
    x = point_undistorted[0]
    y = point_undistorted[1]
    z = 1.0  # frame has z = 1
    frame_point = np.matrix([x, y, z, 1.0]).T

    # Convert frame point in world coordinates
    frame_point = pose.mi @ frame_point
    frame_point = (frame_point / frame_point.item(3))[:-1]

    # Get the slope of the ray passing between focal point and frame in world coordinates
    slope = frame_point - pose.ti

    # Get point of ray that has world z = world_z
    t = (world_z - pose.ti.item(2)) / slope.item(2)
    x_world = t * slope.item(0) + pose.ti.item(0)
    y_world = t * slope.item(1) + pose.ti.item(1)

    world_point = np.array([x_world, y_world, world_z])
    return world_point


# *
def back_projection_distance(img_x: float, img_y: float, distance: float, pose: Pose):
    camera_matrix, distortion = get_info_solvepnp()

    # Get frame point
    point_undistorted = cv.undistortPoints(np.array([img_x, img_y]), camera_matrix, distortion)[0][0]
    x = point_undistorted[0]
    y = point_undistorted[1]
    z = 1.0  # frame has z=1 in camera coordinates
    magnitude = geometric_utility.magnitude(x, y, z)

    # Extend the vector to the given distance
    x = x / magnitude * distance
    y = y / magnitude * distance
    z = z / magnitude * distance
    camera_point = np.matrix([x, y, z, 1.0]).T

    # Convert the point in world coordinates
    world_point = pose.mi @ camera_point
    world_point = np.array(world_point[:-1]).flatten() / world_point.item(3)
    return world_point


# *
def detect_plate_pose(image, center: Tuple[float, float], centers: List[List[float]], debug_img) -> Pose | None:
    circular_marker = CircularMarker()
    export = image.copy()

    # Ellipses indexes - Marker indexes votes
    marker_idx_img_points = [[] for _ in circular_marker.points]
    # Ellipses centers ordered counter-clockwise (like the marker indexes)
    centers = sorted(centers, reverse=True, key=lambda x: geometric_utility.get_angle(center, x))

    # Sliding windows of length 4 to recognize markers
    for i in range(len(centers)):
        # Draw indexes of ellipses
        # cv.putText(export, str(i), [int(centers[i][0]), int(centers[i][1])], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Take 4 contiguous centers
        img_points_temp = [
            centers[i],
            centers[(i + 1) % len(centers)],
            centers[(i + 2) % len(centers)],
            centers[(i + 3) % len(centers)],
        ]

        # Detect the color from the image
        colors = []
        for m in img_points_temp:
            colors.append(MarkerColors.get_from_pixel(image, int(m[0]), int(m[1])))

        # Infer a position from the colors
        marker_idx = circular_marker.get_markers_position(colors)
        if marker_idx is not None:
            # Save votes
            marker_idx_img_points[marker_idx].append(i)
            marker_idx_img_points[(marker_idx + 1) % len(circular_marker.points)].append((i + 1) % len(centers))
            marker_idx_img_points[(marker_idx + 2) % len(circular_marker.points)].append((i + 1) % len(centers))
            marker_idx_img_points[(marker_idx + 3) % len(circular_marker.points)].append((i + 1) % len(centers))

    # Prepare containers for 2D-3D points correspondences
    img_points = []
    dst_points = []
    for i, m in enumerate(marker_idx_img_points):
        if len(m) > 0:
            # Take the most voted marker if it has at least two votes (Can be increased for precision)
            most_frequent = max(set(m), key=m.count)
            if m.count(most_frequent) >= 2:
                dst_points.append(circular_marker.get_markers_point(i))
                img_points.append(centers[most_frequent])
                # Draw indexes of markers
                # cv.putText(export, str(i), centers[most_frequent], cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # export = imutils.resize(export, height=600)
    # cv.imshow("Img_3", export)

    if len(img_points) > 4:  # Can be increased for precision
        # Find homograpy (Used only for debug)
        # M, mask = cv.findHomography(np.array(img_points), np.array([[m[0] + 300, m[1] + 300] for m in dst_points]))
        # img_out = cv.warpPerspective(image, M, (600, 600))
        #
        # img_out = imutils.resize(img_out, height=600)
        # cv.imshow("Img_3", img_out)

        # Solve PnP
        camera_matrix, distortion = get_info_solvepnp()
        success, r, t = cv.solvePnP(np.array(dst_points), np.array(img_points), camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
        if not success:
            return None

        pose = Pose(r, t)

        # Print axis for debug
        zero = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), r, t, camera_matrix, distortion)[0][0][0]
        x_axis = cv.projectPoints(np.array([(100.0, 0.0, 0.0)]), r, t, camera_matrix, distortion)[0][0][0]
        y_axis = cv.projectPoints(np.array([(0.0, 100.0, 0.0)]), r, t, camera_matrix, distortion)[0][0][0]
        z_axis = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), r, t, camera_matrix, distortion)[0][0][0]
        pose_pnt = cv.projectPoints(np.array([pose.ti.item(0), pose.ti.item(1), pose.ti.item(2)]), r, t, camera_matrix, distortion)[0][0][0]
        pose_prj = cv.projectPoints(np.array([pose.ti.item(0), pose.ti.item(1), 0]), r, t, camera_matrix, distortion)[0][0][0]

        cv.line(debug_img, zero.astype('int'), x_axis.astype('int'), (255, 0, 0), 2)
        cv.line(debug_img, zero.astype('int'), y_axis.astype('int'), (0, 255, 0), 2)
        cv.line(debug_img, zero.astype('int'), z_axis.astype('int'), (0, 0, 255), 2)
        cv.line(debug_img, zero.astype('int'), pose_prj.astype('int'), (0, 255, 255), 2)

        return pose

    return None


def ransac(centers: List[List[float]], debug_img):
    n = max(Parameters.ransac_n, 5)

    best_model: Ellipse | None = None
    best_error = 0
    for iteration in range(Parameters.ransac_k):
        possible_inliers_idx = set(random.sample([i for i in range(len(centers))], n))
        possible_inliers = [[int(centers[i][0]), int(centers[i][1])] for i in possible_inliers_idx]
        possible_model = Ellipse(cv.fitEllipse(np.array(possible_inliers)))
        consensus_set_idx = possible_inliers_idx

        for i, c in enumerate(centers):
            if i not in possible_inliers_idx:
                dist = point2ellipse.point_ellipse_distance(possible_model, (c[0], c[1]))
                if dist < Parameters.ransac_t:
                    consensus_set_idx.add(i)

        if len(consensus_set_idx) >= Parameters.ransac_d:
            enhanced_possible_inliers = [[int(centers[i][0]), int(centers[i][1])] for i in consensus_set_idx]
            enhanced_model = Ellipse(cv.fitEllipse(np.array(enhanced_possible_inliers)))
            mean_error = 0
            # max_error = 0
            for i, c in enumerate(centers):
                if i not in consensus_set_idx:
                    dist = point2ellipse.point_ellipse_distance(enhanced_model, (c[0], c[1]))
                    mean_error += dist / (len(centers) - len(consensus_set_idx))
                    # max_error = max(max_error, dist)

            if best_model is None or mean_error < best_error:
                best_model = enhanced_model
                best_error = mean_error

    # Debug print
    cv.ellipse(debug_img, best_model.raw, (0, 255, 255), 2)
    return best_model


# *
def threshold(image, debug_img):
    # Convert to grayscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh_img = cv.threshold(gray_img, Parameters.threshold, 255, 0)

    # Blur
    bilateral_filtered_image = cv.bilateralFilter(thresh_img, 5,  Parameters.sigma, Parameters.sigma)

    # Detect edges
    edge_img = cv.Canny(bilateral_filtered_image, Parameters.a_canny, Parameters.b_canny)

    return edge_img


# *
def detect_wall(edge_img, debug_img) -> Rectangle | None:
    # Apply closing
    MORPH_SIZE = 2
    kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), (MORPH_SIZE, MORPH_SIZE))
    edge_img_rect = cv.morphologyEx(edge_img, cv.MORPH_CLOSE, kernel_rect, iterations=4)

    # Find contours
    contours, hierarchy = cv.findContours(edge_img_rect, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the max rectangle without child
    wall_rect: Rectangle | None = None
    for i, c in enumerate(contours):
        first_child = hierarchy[0][i][2]
        if first_child == -1:
            # Fit a polygon
            points = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
            if len(points) == 4:
                rect = Rectangle(cv.boundingRect(c), points)
                if wall_rect is None or rect.area > wall_rect.area:
                    wall_rect = rect

    # Debug print wall rectangle
    for p in range(len(wall_rect.points)):
        cv.line(debug_img, wall_rect.points[p][0], wall_rect.points[(p + 1) % len(wall_rect.points)][0], (0, 0, 255), 2)

    return wall_rect


# *
def detect_ellipses(edge_img, image, debug_img) -> List[Ellipse | None]:
    # Apply closing
    MORPH_SIZE = 2
    kernel_el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), (MORPH_SIZE, MORPH_SIZE))
    edge_img_el = cv.morphologyEx(edge_img, cv.MORPH_CLOSE, kernel_el, iterations=4)

    # Find contours
    contours, hierarchy = cv.findContours(edge_img_el, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the ellipses for each contour
    ellipses: List[Ellipse | None] = [None] * len(contours)
    for i, c in enumerate(contours):
        # Filter by number of points
        if Parameters.ellipses_min_points < c.shape[0] < Parameters.ellipses_max_points:
            ellipses[i] = Ellipse(cv.fitEllipse(c))

    # Filter out wrong ellipses
    for i, e in enumerate(ellipses):
        if e is not None:
            # Filter by ellipse ratio
            if e.max_size > e.min_size * (Parameters.ellipses_ratio / 100):
                ellipses[i] = None
            else:
                # Calculate a shape error estimation
                mean_err = 0
                for p in contours[i]:
                    err = point2ellipse.point_ellipse_distance(e, (p[0][0], p[0][1]))
                    mean_err += err / len(contours[i])
                if mean_err > Parameters.ellipses_precision:
                    ellipses[i] = None

    # Filter ellipses too near to another ellipses
    for i, e in enumerate(ellipses):
        if e is not None:
            center = e.center
            size = e.x_size * e.y_size
            for i2, e2 in enumerate(ellipses):
                if e2 is not None and i != i2:
                    center2 = e2.center
                    # if distance is less than max axes
                    if np.sqrt(((center.x - center2.x) ** 2) + ((center.y - center2.y) ** 2)) < e2.max_size:
                        size2 = e2.x_size * e2.y_size
                        if size2 > size:  # Keep the largest one
                            ellipses[i] = None

    # Remove Nones from ellipses
    ellipses = [e for e in ellipses if e is not None]

    # Debug print
    for ellipse in ellipses:
        cv.ellipse(debug_img, ellipse.raw, MarkerColors.get_from_pixel_debug(image, int(ellipse.center.x), int(ellipse.center.y)), 3)

    return ellipses


# *
def order_wall_points(wall_rect: Rectangle) -> Rectangle:
    # Separate points in top and bottom and then order by x
    unpacked_points = [i[0] * 1.0 for i in wall_rect.points]
    points_y_sorted = sorted(unpacked_points, key=lambda y: y[1])
    top_points = sorted(points_y_sorted[:2], key=lambda x: x[0])
    bottom_points = sorted(points_y_sorted[2:], key=lambda x: x[0])

    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    wall_rect.points = top_points + bottom_points
    return wall_rect


# *
def detect_wall_pose(wall_rect: Rectangle, debug_img) -> Pose | None:
    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = wall_rect.points

    # SolvePnP
    camera_matrix, distortion = get_info_solvepnp()
    success, r, t = cv.solvePnP(np.array(WallMarker.points), np.array(img_points), camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
    if not success:
        return None

    # Print axis for debug
    zero = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), r, t, camera_matrix, distortion)[0][0][0]
    x_axis = cv.projectPoints(np.array([(100.0, 0.0, 0.0)]), r, t, camera_matrix, distortion)[0][0][0]
    y_axis = cv.projectPoints(np.array([(0.0, 100.0, 0.0)]), r, t, camera_matrix, distortion)[0][0][0]
    z_axis = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), r, t, camera_matrix, distortion)[0][0][0]
    cv.line(debug_img, zero .astype('int'), x_axis .astype('int'), (255, 0, 0), 2)
    cv.line(debug_img, zero .astype('int'), y_axis .astype('int'), (0, 255, 0), 2)
    cv.line(debug_img, zero .astype('int'), z_axis .astype('int'), (0, 0, 255), 2)

    return Pose(r, t)


# *
def get_3d_wall_corners(wall_rect: Rectangle, plate_pose: Pose, wall_pose: Pose, debug_img) -> list[np.ndarray]:
    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = wall_rect.points

    wall_3d_points: list[np.ndarray] = []
    for i in range(4):
        # Get distance of point from focal point
        distance_from_point = geometric_utility.magnitude(wall_pose.ti.item(0) - WallMarker.points[i][0], wall_pose.ti.item(1) - WallMarker.points[i][1], wall_pose.ti.item(2))
        # Get wall points relative to plate_zero
        wall_3d_points.append(back_projection_distance(img_points[i][0], img_points[i][1], distance_from_point, plate_pose))

    # Print coordinates for debug
    cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[0][0]), int(wall_3d_points[0][1]), int(wall_3d_points[0][2])), [50, 100], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[1][0]), int(wall_3d_points[1][1]), int(wall_3d_points[1][2])), [600, 100], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[2][0]), int(wall_3d_points[2][1]), int(wall_3d_points[2][2])), [50, 1000], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[3][0]), int(wall_3d_points[3][1]), int(wall_3d_points[3][2])), [600, 1000], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return wall_3d_points


# *
def detect_wall_line(image, wall_rect: Rectangle, wall_3d_points: list[np.ndarray], plate_pose: Pose, debug_img):
    margin = 5  # Prevent black pixel of contours

    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = wall_rect.points

    # Get homography of wall marker
    M, mask = cv.findHomography(np.array(img_points), np.array(WallMarker.points))
    homo_img = cv.warpPerspective(image, M, (WallMarker.w, WallMarker.h))

    # Filter for red line
    lower_red = np.array([Parameters.laser_b_dw, Parameters.laser_g_dw, Parameters.laser_r_dw], dtype="uint8")
    upper_red = np.array([Parameters.laser_b_up, Parameters.laser_g_up, Parameters.laser_r_up], dtype="uint8")
    mask = cv.inRange(homo_img, lower_red, upper_red)
    # edge_img = cv.bitwise_and(homo_img, homo_img, mask=mask)

    # Get a list of point in line
    points_of_line = []
    for x in range(margin, WallMarker.w - margin, LASER_PLANE_FIT_INTERVAL_X):
        for y in range(margin, WallMarker.h - margin, LASER_PLANE_FIT_INTERVAL_Y):
            if mask[y][x] == 255:
                points_of_line.append([[x, y]])

    # Not enough points
    if len(points_of_line) < 5:
        return None, None

    # Fit line of laser on the wall
    line = cv.fitLine(np.array(points_of_line), cv.DIST_L2, 0, 0.01, 0.01)
    vx = line[0][0]
    vy = line[1][0]
    x0 = line[2][0]
    y0 = line[3][0]

    # Print line for debug
    t = 200
    cv.line(homo_img, np.array([int(x0 - t * vx), int(y0 - t * vy)]), np.array([int(x0 + t * vx), int(y0 + t * vy)]), (255, 255, 0))

    # Get intersection of line with top and bottom line of marker
    y1 = 0
    y2 = WallMarker.h_dec
    t1 = (y1 - y0)/vy
    t2 = (y2 - y0)/vy
    x1 = x0 + t1 * vx
    x2 = x0 + t2 * vx

    # Print intersection points for debug
    cv.circle(homo_img, [int(x1), int(y1)], 2, (0, 255, 0), cv.FILLED)
    cv.circle(homo_img, [int(x2), int(y2)], 2, (0, 255, 0), cv.FILLED)

    homo_img = imutils.resize(homo_img, height=600)
    cv.imshow("Img_3", homo_img)

    # Get positions relative to the segments
    top_point_percentage = x1 / WallMarker.w_dec
    bottom_point_percentage = x2 / WallMarker.w_dec

    # Get 3d points
    top_3d_diff = wall_3d_points[1] - wall_3d_points[0]
    bottom_3d_diff = wall_3d_points[3] - wall_3d_points[2]
    top_3d_point = (top_3d_diff * top_point_percentage) + wall_3d_points[0]
    bottom_3d_point = (bottom_3d_diff * bottom_point_percentage) + wall_3d_points[2]

    # Debug print
    camera_matrix, distortion = get_info_solvepnp()
    p1 = cv.projectPoints(top_3d_point, plate_pose.r, plate_pose.t, camera_matrix, distortion)[0][0][0]
    p2 = cv.projectPoints(bottom_3d_point, plate_pose.r, plate_pose.t, camera_matrix, distortion)[0][0][0]
    cv.circle(debug_img, p1.astype('int'), 10, (0, 255, 0), cv.FILLED)
    cv.circle(debug_img, p2.astype('int'), 10, (0, 255, 0), cv.FILLED)

    return top_3d_point, bottom_3d_point


# *
def detect_plate_laser_point(hsv_img, plate_pose: Pose, debug_img):
    camera_matrix, distortion = get_info_solvepnp()

    # Get search area (based on the pose)
    search_area_dst_points = CircularMarker.get_laser_search_area(plate_pose.ti.item(0), plate_pose.ti.item(1))
    # Project search area on frame
    search_area_img_points = [cv.projectPoints(np.array(i), plate_pose.r, plate_pose.t, camera_matrix, distortion)[0][0][0] for i in search_area_dst_points]

    def print_search_area(img):
        cv.line(img, search_area_img_points[0].astype('int'), search_area_img_points[1].astype('int'), (0, 255, 255), 3)
        cv.line(img, search_area_img_points[1].astype('int'), search_area_img_points[3].astype('int'), (0, 255, 255), 3)
        cv.line(img, search_area_img_points[3].astype('int'), search_area_img_points[2].astype('int'), (0, 255, 255), 3)
        cv.line(img, search_area_img_points[2].astype('int'), search_area_img_points[0].astype('int'), (0, 255, 255), 3)

    # Filter for red line
    lower_red = np.array([Parameters.laser_h_dw, Parameters.laser_s_dw, Parameters.laser_v_dw], dtype="int")
    upper_red = np.array([Parameters.laser_h_up, Parameters.laser_s_up, Parameters.laser_v_up], dtype="int")
    mask = cv.inRange(hsv_img, lower_red, upper_red)

    min_x = min([int(i[0]) for i in search_area_img_points])
    max_x = max([int(i[0]) for i in search_area_img_points])
    min_y = min([int(i[1]) for i in search_area_img_points])
    max_y = max([int(i[1]) for i in search_area_img_points])

    # Get a list of point in line
    points_of_line = []
    for x in range(min_x, max_x, LASER_PLANE_FIT_INTERVAL_X):
        for y in range(min_y, max_y, LASER_PLANE_FIT_INTERVAL_Y):
            if mask[y][x] == 255 and geometric_utility.is_inside_polygon(np.array([[x, y]]), np.array([search_area_img_points[0], search_area_img_points[1], search_area_img_points[3], search_area_img_points[2], search_area_img_points[0]])).shape[0] > 0:
                points_of_line.append([[x, y]])

    # Not enough points
    if len(points_of_line) < 5:
        return None

    # Fit line of laser on the plate
    line = cv.fitLine(np.array(points_of_line), cv.DIST_L2, 0, 0.01, 0.01)
    # vx: float = line[0][0]
    # vy: float = line[1][0]
    x0: float = line[2][0]
    y0: float = line[3][0]

    cv.circle(debug_img, np.array([x0, y0]).astype('int'), 10, (0, 255, 0), cv.FILLED)

    print_search_area(debug_img)
    # img_out = cv.bitwise_and(image, image, mask=mask)
    # print_search_area(img_out)
    # img_out = imutils.resize(img_out, height=600)
    # cv.imshow("Img_4", img_out)

    intersection_3d = back_projection_z(x0, y0, 0.0, plate_pose)
    return intersection_3d


# *
def detect_object_points(hsv_img, out_file: OutputXYZ, plate_pose: Pose, laser_plane, debug_img):
    camera_matrix, distortion = get_info_solvepnp()

    # Define search area
    zero = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), plate_pose.r, plate_pose.t, camera_matrix, distortion)[0][0][0].astype('int')
    min_x = -100
    max_x = 120
    min_y = -250
    max_y = 120
    p1 = zero + np.array([min_x, min_y])
    p2 = zero + np.array([max_x, min_y])
    p3 = zero + np.array([max_x, max_y])
    p4 = zero + np.array([min_x, max_y])

    # Filter for red line
    lower_red = np.array([Parameters.laser_h_dw, Parameters.laser_s_dw, Parameters.laser_v_dw], dtype="int")
    upper_red = np.array([Parameters.laser_h_up, Parameters.laser_s_up, Parameters.laser_v_up], dtype="int")
    mask = cv.inRange(hsv_img, lower_red, upper_red)

    # Iterate all red pixels
    for x in range(zero[0] + min_x, zero[0] + max_x, SCANNER_INTERVAL):
        for y in range(zero[1] + min_y, zero[1] + max_y, SCANNER_INTERVAL):
            if mask[y][x] == 255:
                # Save 3d object point
                obj_3d_point = back_projection_plane(float(x), float(y), laser_plane, plate_pose)
                if obj_3d_point is not None:
                    out_file.add_point(obj_3d_point)

    def print_search_area(img):
        cv.line(img, p1, p2, (255, 0, 255), 2)
        cv.line(img, p2, p3, (255, 0, 255), 2)
        cv.line(img, p3, p4, (255, 0, 255), 2)
        cv.line(img, p4, p1, (255, 0, 255), 2)

    # Debug Print
    img_out = cv.bitwise_and(hsv_img, hsv_img, mask=mask)
    print_search_area(debug_img)
    print_search_area(img_out)
    img_out = imutils.resize(img_out, height=600)
    cv.imshow("Img_4", img_out)


def main():
    cv.namedWindow("Img_1")
    cv.namedWindow("Img_2")
    cv.namedWindow("Img_3")
    cv.namedWindow("Img_4")

    parameters.init_parameters()

    video = cv.VideoCapture('.\\data\\ball.mov')

    out_file = OutputXYZ()
    start_time = time.time()

    # Loop the frames in the video and take NUM_OF_FRAMES equally spaced frames
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_i in range(video_length):
        frame_time = time.time() - start_time
        start_time = time.time()
        fps = 1/frame_time if frame_time != 0 else 0

        success, image = video.read()
        if not success:
            continue

        image = undistort_image(image)
        debug_img = image.copy()
        drop_img = image.copy()

        parameters.update_parameters()

        cv.putText(debug_img, "fps: {:.2f}".format(fps), [50, 50], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        edge_img = threshold(image, debug_img)

        # Wall
        wall_rect = detect_wall(edge_img, debug_img)
        wall_rect = order_wall_points(wall_rect)
        wall_pose = detect_wall_pose(wall_rect, debug_img)
        if wall_pose is None:
            continue

        # Plate ellipses
        ellipses = detect_ellipses(edge_img, image, debug_img)
        if len(ellipses) < 5:
            continue

        ellipses_centers = []
        ellipses_centers_int = []
        for ellipse in ellipses:
            ellipses_centers.append([ellipse.center.x, ellipse.center.y])
            ellipses_centers_int.append([int(ellipse.center.x), int(ellipse.center.y)])

        # An ellipse to rule them all
        ellipse_master = ransac(ellipses_centers, debug_img)
        if ellipse_master is None:
            continue

        plate_pose = detect_plate_pose(image, ellipse_master.center.raw, ellipses_centers, debug_img)
        if plate_pose is None:
            continue

        plate_3d_point = detect_plate_laser_point(hsv_img, plate_pose, debug_img)
        if plate_3d_point is None:
            continue

        wall_3d_points = get_3d_wall_corners(wall_rect, plate_pose, wall_pose, drop_img)

        top_3d_point, bottom_3d_point = detect_wall_line(image, wall_rect, wall_3d_points, plate_pose, debug_img)
        if top_3d_point is None or bottom_3d_point is None:
            continue

        laser_plane = geometric_utility.get_plane(top_3d_point, bottom_3d_point, plate_3d_point)

        detect_object_points(hsv_img, out_file, plate_pose, laser_plane, debug_img)

        debug_img = imutils.resize(debug_img, height=600)
        edge_img = imutils.resize(edge_img, height=600)
        cv.imshow("Img_1", debug_img)
        cv.imshow("Img_2", edge_img)

        # ESC to break
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
    pass
