import random
from typing import List, Tuple
import point2ellipse
import geometric_utility
from circular_marker import CircularMarker, MarkerColors
import numpy as np
import cv2 as cv
import imutils

from geometry import Rectangle, Ellipse
from pose import Pose
from undistorion import undistort_image, get_new_camera_matrix, get_distortion, get_h_w
from wall_marker import WallMarker


def nothing(x):
    pass


def get_info_solvepnp():
    new_camera_matrix = get_new_camera_matrix()
    distortion = np.zeros((4, 1))  # get_distortion()
    return new_camera_matrix, distortion


def back_projection_z(img_x, img_y, z: float, pose: Pose):
    new_camera_matrix, distortion = get_info_solvepnp()

    point_undistorted = cv.undistortPoints(np.array([img_x, img_y]), new_camera_matrix, distortion)
    x = point_undistorted[0][0][0]
    y = point_undistorted[0][0][1]
    z = 1
    frame_point = np.matrix([x, y, z, 1]).T
    frame_point = pose.mi @ frame_point
    frame_point = frame_point / frame_point.item(3)

    vx = frame_point.item(0) - pose.ti.item(0)
    vy = frame_point.item(1) - pose.ti.item(1)
    vz = frame_point.item(2) - pose.ti.item(2)

    z_world = 0
    t = (z_world - pose.ti.item(2)) / vz
    x_world = t * vx + pose.ti.item(0)
    y_world = t * vy + pose.ti.item(1)

    world_point = np.array([x_world, y_world, z_world])
    return world_point


def back_projection_distance(img_x, img_y, distance: float, pose: Pose):
    new_camera_matrix, distortion = get_info_solvepnp()

    point_undistorted = cv.undistortPoints(np.array([img_x, img_y]), new_camera_matrix, distortion)
    x = point_undistorted[0][0][0]
    y = point_undistorted[0][0][1]
    z = 1
    magnitude = geometric_utility.magnitude(x, y, z)

    x = x / magnitude * distance
    y = y / magnitude * distance
    z = z / magnitude * distance
    camera_point = np.matrix([x, y, z, 1]).T
    world_point = pose.mi @ camera_point
    world_point = np.array([
        world_point.item(0) / world_point.item(3),
        world_point.item(1) / world_point.item(3),
        world_point.item(2) / world_point.item(3),
    ])
    return world_point


def detect_plate_pose(image, center: Tuple[float, float], centers: List[List[float]], debug_img) -> Pose:
    circular_marker = CircularMarker()
    export = image.copy()

    img_points = []
    dst_points = []
    marker_idx_img_points = [[] for _ in circular_marker.points]
    centers = sorted(centers, reverse=True, key=lambda x: geometric_utility.get_angle(center, x))

    i = 0
    while i < len(centers):
        export = cv.putText(export, str(i), [int(centers[i][0]), int(centers[i][1])], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img_points_temp = [
            centers[i],
            centers[(i + 1) % len(centers)],
            centers[(i + 2) % len(centers)],
            centers[(i + 3) % len(centers)],
        ]

        colors = []
        for m in img_points_temp:
            colors.append(MarkerColors.get_from_pixel(image, int(m[0]), int(m[1])))

        marker_idx = circular_marker.get_markers_position(colors)
        if marker_idx is not None:
            dst_points_temp = circular_marker.get_markers_points(marker_idx)
            img_points.extend(img_points_temp)
            dst_points.extend(dst_points_temp)
            marker_idx_img_points[marker_idx].append(i)
            marker_idx_img_points[(marker_idx + 1) % len(circular_marker.points)].append((i + 1) % len(centers))
            marker_idx_img_points[(marker_idx + 2) % len(circular_marker.points)].append((i + 1) % len(centers))
            marker_idx_img_points[(marker_idx + 3) % len(circular_marker.points)].append((i + 1) % len(centers))
        #     i += 4
        # else:
        i += 1

    img_points = []
    dst_points = []
    for i, m in enumerate(marker_idx_img_points):
        if len(m) > 0:
            most_frequent = max(set(m), key=m.count)
            if m.count(most_frequent) > 1:
                dst_points.append(circular_marker.get_markers_point(i))
                img_points.append(centers[most_frequent])
                # export = cv.putText(export, str(i), centers[most_frequent], cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # export = imutils.resize(export, height=600)
    # cv.imshow("Img_3", export)

    if len(img_points) > 4:  # TODO pensare di aumentare per precisione
        # Find homograpy
        M, mask = cv.findHomography(np.array(img_points), np.array([[m[0] + 300, m[1] + 300] for m in dst_points]))
        img_out = cv.warpPerspective(image, M, (600, 600))

        img_out = imutils.resize(img_out, height=600)
        # cv.imshow("Img_3", img_out)

        new_camera_matrix, distortion = get_info_solvepnp()
        success, r, t = cv.solvePnP(np.array(dst_points), np.array(img_points), new_camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
        if not success:
            return None

        pose = Pose(r, t)
        # cv.undistortPoint()
        # test_point, _ = cv.projectPoints(np.array(circular_marker.get_marker_point(marker_idx-1)), r, t, new_camera_matrix, distortion)
        # test_point_x = int(test_point[0][0][0])
        # test_point_y = int(test_point[0][0][1])
        # zero_point, _ = cv.projectPoints(np.array([0.0, 0.0, 0.0]), r, t, new_camera_matrix, distortion)
        # zero_point_x = int(zero_point[0][0][0])
        # zero_point_y = int(zero_point[0][0][1])
        # cv.line(export, (0, 0), (zero_point_x, zero_point_y), (0, 0, 0), 4)
        # cv.imwrite(".\\data\\debug\\image.jpg", export)

        # if 0 <= test_point_x < w and 0 <= test_point_y < h and \
        #    MarkerColors.get_from_pixel(image[test_point_y][test_point_x]) == circular_marker.get_marker_color(marker_idx-1):
        zero, _ = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), r, t, new_camera_matrix, distortion)
        x_axis, _ = cv.projectPoints(np.array([(100.0, 0.0, 0.0)]), r, t, new_camera_matrix, distortion)
        y_axis, _ = cv.projectPoints(np.array([(0.0, 100.0, 0.0)]), r, t, new_camera_matrix, distortion)
        z_axis, _ = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), r, t, new_camera_matrix, distortion)
        pose_pnt, _ = cv.projectPoints(np.array([pose.ti.item(0), pose.ti.item(1), pose.ti.item(2)]), r, t, new_camera_matrix, distortion)
        pose_prj, _ = cv.projectPoints(np.array([pose.ti.item(0), pose.ti.item(1), 0]), r, t, new_camera_matrix, distortion)

        cv.line(debug_img, zero[0][0].astype('int'), x_axis[0][0].astype('int'), (255, 0, 0), 2)
        cv.line(debug_img, zero[0][0].astype('int'), y_axis[0][0].astype('int'), (0, 255, 0), 2)
        cv.line(debug_img, zero[0][0].astype('int'), z_axis[0][0].astype('int'), (0, 0, 255), 2)
        cv.line(debug_img, zero[0][0].astype('int'), pose_prj[0][0].astype('int'), (0, 255, 255), 2)

        return pose

    return None, None


def ransac(centers: List[List[float]]):
    n = cv.getTrackbarPos('n', 'Ransac')
    k = cv.getTrackbarPos('k', 'Ransac')
    t = cv.getTrackbarPos('t', 'Ransac')
    d = cv.getTrackbarPos('d', 'Ransac')

    n = max(n, 5)

    best_model: Ellipse | None = None
    best_consensus_set = None
    best_error = 0
    for iteration in range(k):
        possible_inliers_idx = set(random.sample([i for i in range(len(centers))], n))
        possible_inliers = [[int(centers[i][0]), int(centers[i][1])] for i in possible_inliers_idx]
        possible_model = Ellipse(cv.fitEllipse(np.array(possible_inliers)))
        consensus_set_idx = possible_inliers_idx

        for i, c in enumerate(centers):
            if i not in possible_inliers_idx:
                dist = point2ellipse.point_ellipse_distance(possible_model, (c[0], c[1]))
                if dist < t:
                    consensus_set_idx.add(i)

        if len(consensus_set_idx) >= d:
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
                best_consensus_set = consensus_set_idx
                best_error = mean_error

    return best_model


def threshold(image, debug_img):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    threshold = cv.getTrackbarPos('threshold', 'Parameters')
    sigma = cv.getTrackbarPos('sigma', 'Parameters')
    a_canny = cv.getTrackbarPos('a_canny', 'Parameters')
    b_canny = cv.getTrackbarPos('b_canny', 'Parameters')
    ret, thresh_img = cv.threshold(gray_img, threshold, 255, 0)

    # Blur an image
    bilateral_filtered_image = cv.bilateralFilter(thresh_img, 5, sigma, sigma)
    # Detect edges
    edge_img = cv.Canny(bilateral_filtered_image, a_canny, b_canny)

    return edge_img


MORPH_SIZE = 2


def detect_wall(edge_img, debug_img) -> Rectangle | None:
    kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), (MORPH_SIZE, MORPH_SIZE))
    edge_img_rect = cv.morphologyEx(edge_img, cv.MORPH_CLOSE, kernel_rect, iterations=4)

    # Find contours
    contours, hierarchy = cv.findContours(edge_img_rect, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the max rectangle without child
    wall_rect: Rectangle | None = None
    for i, c in enumerate(contours):
        points = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
        first_child = hierarchy[0][i][2]
        if len(points) == 4 and first_child == -1:
            rect = Rectangle(cv.boundingRect(c), points)
            if wall_rect is None or rect.area > wall_rect.area:
                wall_rect = rect

    # Debug print rectangle
    for p in range(len(wall_rect.points)):
        debug_img = cv.line(debug_img, wall_rect.points[p][0], wall_rect.points[(p + 1) % len(wall_rect.points)][0], (0, 0, 255), 2)

    return wall_rect


def detect_ellipses(edge_img, image, debug_img) -> List[Ellipse | None]:
    ellipses_precision = cv.getTrackbarPos('ellipses_precision', 'Parameters')
    ellipses_min_points = cv.getTrackbarPos('ellipses_min_points', 'Parameters')
    ellipses_max_points = cv.getTrackbarPos('ellipses_max_points', 'Parameters')
    ellipses_ratio = cv.getTrackbarPos('ellipses_ratio', 'Parameters')

    kernel_el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), (MORPH_SIZE, MORPH_SIZE))
    edge_img_el = cv.morphologyEx(edge_img, cv.MORPH_CLOSE, kernel_el, iterations=4)

    # Find contours
    contours, hierarchy = cv.findContours(edge_img_el, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the ellipses for each contour
    ellipses: List[Ellipse | None] = [None] * len(contours)
    for i, c in enumerate(contours):
        if ellipses_min_points < c.shape[0] < ellipses_max_points:
            # rects[i] = cv.minAreaRect(c)
            ellipses[i] = Ellipse(cv.fitEllipse(c))

    # Filter wrong ellipses
    for i, e in enumerate(ellipses):
        if e is not None:
            if e.max_size > e.min_size * (ellipses_ratio / 100):
                ellipses[i] = None
            else:
                # Calculate a shape error estimation
                mean_err = 0
                for p in contours[i]:
                    err = point2ellipse.point_ellipse_distance(e, (p[0][0], p[0][1]))
                    mean_err += err / len(contours[i])
                if mean_err > ellipses_precision:
                    ellipses[i] = None

    # Filter ellipses too near
    for i, e in enumerate(ellipses):
        if e is not None:
            center = e.center
            size = e.x_size * e.y_size

            for i2, e2 in enumerate(ellipses):
                if e2 is not None and i != i2:
                    center2 = e2.center
                    if np.sqrt(((center.x - center2.x) ** 2) + ((center.y - center2.y) ** 2)) < e2.max_size:
                        size2 = e2.x_size * e2.y_size
                        if size2 > size:
                            ellipses[i] = None

    # Remove Nones from ellipses
    ellipses = [e for e in ellipses if e is not None]

    # Debug print
    for ellipse in ellipses:
        cv.ellipse(debug_img, ellipse.raw, MarkerColors.get_from_pixel_debug(image, int(ellipse.center.x), int(ellipse.center.y)), 3)

    return ellipses


def order_wall_points(wall_rect: Rectangle) -> Rectangle:
    unpacked_points = [i[0] * 1.0 for i in wall_rect.points]
    points_y_sorted = sorted(unpacked_points, key=lambda x: x[1])
    top_points = sorted(points_y_sorted[:2], key=lambda x: x[0])
    bottom_points = sorted(points_y_sorted[2:], key=lambda x: x[0])

    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    wall_rect.points = top_points + bottom_points
    return wall_rect


def detect_wall_pose(image, wall_rect: Rectangle, debug_img) -> Pose:
    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = wall_rect.points

    new_camera_matrix, distortion = get_info_solvepnp()
    success, r, t = cv.solvePnP(np.array(WallMarker.points), np.array(img_points), new_camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
    if not success:
        return None

    zero, _ = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), r, t, new_camera_matrix, distortion)
    x_axis, _ = cv.projectPoints(np.array([(100.0, 0.0, 0.0)]), r, t, new_camera_matrix, distortion)
    y_axis, _ = cv.projectPoints(np.array([(0.0, 100.0, 0.0)]), r, t, new_camera_matrix, distortion)
    z_axis, _ = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), r, t, new_camera_matrix, distortion)

    cv.line(debug_img, zero[0][0].astype('int'), x_axis[0][0].astype('int'), (255, 0, 0), 2)
    cv.line(debug_img, zero[0][0].astype('int'), y_axis[0][0].astype('int'), (0, 255, 0), 2)
    cv.line(debug_img, zero[0][0].astype('int'), z_axis[0][0].astype('int'), (0, 0, 255), 2)

    return Pose(r, t)


def get_3d_wall_points(wall_rect: Rectangle, plate_pose: Pose, wall_pose: Pose, debug_img) -> list[np.ndarray]:
    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = wall_rect.points

    wall_3d_points: list[np.ndarray] = []
    for i in range(4):
        distance_from_point = geometric_utility.magnitude(wall_pose.ti.item(0) - WallMarker.points[i][0], wall_pose.ti.item(1) - WallMarker.points[i][1], wall_pose.ti.item(2))
        wall_3d_points.append(back_projection_distance(img_points[i][0], img_points[i][1], distance_from_point, plate_pose))

    debug_img = cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[0][0]), int(wall_3d_points[0][1]), int(wall_3d_points[0][2])), [50, 100], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    debug_img = cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[1][0]), int(wall_3d_points[1][1]), int(wall_3d_points[1][2])), [600, 100], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    debug_img = cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[2][0]), int(wall_3d_points[2][1]), int(wall_3d_points[2][2])), [50, 1000], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    debug_img = cv.putText(debug_img, "{}, {}, {}".format(int(wall_3d_points[3][0]), int(wall_3d_points[3][1]), int(wall_3d_points[3][2])), [600, 1000], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return wall_3d_points


def detect_wall_line(image, wall_rect: Rectangle, wall_3d_points: list[np.ndarray], debug_img):
    margin = 5  # Prevent black pixel

    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = wall_rect.points

    M, mask = cv.findHomography(np.array(img_points), np.array(WallMarker.points))
    homo_img = cv.warpPerspective(image, M, (WallMarker.w, WallMarker.h))

    # lower_red = np.array([170, 170, 205], dtype="uint8")
    # upper_red = np.array([255, 255, 255], dtype="uint8")
    lower_red = np.array([150, 150, 205], dtype="uint8")
    upper_red = np.array([195, 195, 255], dtype="uint8")
    mask = cv.inRange(homo_img, lower_red, upper_red)
    # edge_img = cv.bitwise_and(homo_img, homo_img, mask=mask)

    points_of_line = []
    for x in range(margin, WallMarker.w - margin):
        for y in range(margin, WallMarker.h - margin):
            if mask[y][x] == 255:
                points_of_line.append([[x, y]])

    if len(points_of_line) < 5:
        return None, None
    line = cv.fitLine(np.array(points_of_line), cv.DIST_L2, 0, 0.01, 0.01)
    vx = line[0][0]
    vy = line[1][0]
    x0 = line[2][0]
    y0 = line[3][0]
    t = 200
    cv.line(homo_img, np.array([int(x0 - t * vx), int(y0 - t * vy)]), np.array([int(x0 + t * vx), int(y0 + t * vy)]), (255, 255, 0))

    y1 = 0
    y2 = WallMarker.h_dec
    t1 = (y1 - y0)/vy
    t2 = (y2 - y0)/vy
    x1 = x0 + t1 * vx
    x2 = x0 + t2 * vx

    cv.circle(homo_img, [int(x1), int(y1)], 2, (0, 255, 0), cv.FILLED)
    cv.circle(homo_img, [int(x2), int(y2)], 2, (0, 255, 0), cv.FILLED)

    homo_img = imutils.resize(homo_img, height=600)
    cv.imshow("Img_3", homo_img)

    top_point_percentage = x1 / WallMarker.w_dec
    bottom_point_percentage = x2 / WallMarker.w_dec

    top_3d_diff = wall_3d_points[1] - wall_3d_points[0]
    bottom_3d_diff = wall_3d_points[3] - wall_3d_points[2]
    top_3d_point = (top_3d_diff * top_point_percentage) + wall_3d_points[0]
    bottom_3d_point = (bottom_3d_diff * bottom_point_percentage) + wall_3d_points[2]

    return top_3d_point, bottom_3d_point


def detect_plate_laser_point(image, plate_pose: Pose, debug_img):
    new_camera_matrix, distortion = get_info_solvepnp()
    search_area_dst_points = CircularMarker.get_laser_search_area(plate_pose.ti.item(0), plate_pose.ti.item(1))
    search_area_img_points = [cv.projectPoints(np.array(i), plate_pose.r, plate_pose.t, new_camera_matrix, distortion)[0][0][0] for i in search_area_dst_points]

    def print_search_area(img):
        cv.line(img, search_area_img_points[0].astype('int'), search_area_img_points[1].astype('int'), (0, 255, 255), 3)
        cv.line(img, search_area_img_points[1].astype('int'), search_area_img_points[3].astype('int'), (0, 255, 255), 3)
        cv.line(img, search_area_img_points[3].astype('int'), search_area_img_points[2].astype('int'), (0, 255, 255), 3)
        cv.line(img, search_area_img_points[2].astype('int'), search_area_img_points[0].astype('int'), (0, 255, 255), 3)

    r_dw = cv.getTrackbarPos('r_dw', 'Red_Filter')
    r_up = cv.getTrackbarPos('r_up', 'Red_Filter')
    g_dw = cv.getTrackbarPos('g_dw', 'Red_Filter')
    g_up = cv.getTrackbarPos('g_up', 'Red_Filter')
    b_dw = cv.getTrackbarPos('b_dw', 'Red_Filter')
    b_up = cv.getTrackbarPos('b_up', 'Red_Filter')

    lower_red = np.array([130, 130, 225], dtype="uint8")
    upper_red = np.array([200, 200, 255], dtype="uint8")
    lower_red = np.array([b_dw, g_dw, r_dw], dtype="uint8")
    upper_red = np.array([b_up, g_up, r_up], dtype="uint8")
    mask = cv.inRange(image, lower_red, upper_red)
    img_out = cv.bitwise_and(image, image, mask=mask)

    min_x = min([int(i[0]) for i in search_area_img_points])
    max_x = max([int(i[0]) for i in search_area_img_points])
    min_y = min([int(i[1]) for i in search_area_img_points])
    max_y = max([int(i[1]) for i in search_area_img_points])

    points_of_line = []
    for x in range(min_x, max_x):  # TODO check if inside polygon
        for y in range(min_y, max_y):
            if mask[y][x] == 255:
                points_of_line.append([[x, y]])

    if len(points_of_line) < 5:
        return None, None
    line = cv.fitLine(np.array(points_of_line), cv.DIST_L2, 0, 0.01, 0.01)
    vx: float = line[0][0]
    vy: float = line[1][0]
    x0: float = line[2][0]
    y0: float = line[3][0]
    m = 100
    cv.line(img_out, np.array([int(x0 - m * vx), int(y0 - m * vy)]), np.array([int(x0 + m * vx), int(y0 + m * vy)]), (255, 255, 0), 3)

    intersection_2d = geometric_utility.get_line_intersection([x0, y0], [x0 + vx, y0 + vy], search_area_img_points[2], search_area_img_points[3])
    cv.circle(debug_img, (intersection_2d).astype('int'), 10, (0, 255, 0), cv.FILLED)

    print_search_area(debug_img)
    print_search_area(img_out)
    img_out = imutils.resize(img_out, height=600)
    cv.imshow("Img_4", img_out)

    intersection_3d = back_projection_z(intersection_2d[0], intersection_2d[1], 0.0, plate_pose)
    return intersection_3d


def main():
    cv.namedWindow("Img_1")
    cv.namedWindow("Img_2")
    cv.namedWindow("Img_3")
    cv.namedWindow("Img_4")
    cv.namedWindow("Parameters")
    cv.createTrackbar('threshold', 'Parameters', 87, 255, nothing)
    cv.createTrackbar('ellipses_precision', 'Parameters', 30, 100, nothing)
    cv.createTrackbar('ellipses_min_points', 'Parameters', 30, 100, nothing)
    cv.createTrackbar('ellipses_max_points', 'Parameters', 300, 1000, nothing)
    cv.createTrackbar('ellipses_ratio', 'Parameters', 200, 1000, nothing)
    cv.createTrackbar('sigma', 'Parameters', 150, 255, nothing)
    cv.createTrackbar('a_canny', 'Parameters', 75, 255, nothing)
    cv.createTrackbar('b_canny', 'Parameters', 200, 255, nothing)

    cv.namedWindow("Ransac")
    cv.createTrackbar('n', 'Ransac', 5, 30, nothing)
    cv.createTrackbar('k', 'Ransac', 400, 5000, nothing)
    cv.createTrackbar('t', 'Ransac', 400, 400, nothing)  # 25
    cv.createTrackbar('d', 'Ransac', 7, 30, nothing)

    cv.namedWindow("Red_Filter")
    cv.createTrackbar('b_dw', 'Red_Filter', 110, 255, nothing)
    cv.createTrackbar('b_up', 'Red_Filter', 200, 255, nothing)
    cv.createTrackbar('g_dw', 'Red_Filter', 110, 255, nothing)
    cv.createTrackbar('g_up', 'Red_Filter', 200, 255, nothing)
    cv.createTrackbar('r_dw', 'Red_Filter', 215, 255, nothing)
    cv.createTrackbar('r_up', 'Red_Filter', 255, 255, nothing)

    video = cv.VideoCapture('.\\data\\cube.mov')

    # Loop the frames in the video and take NUM_OF_FRAMES equally spaced frames
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_i in range(video_length):
        success, image = video.read()
        if success:
            image = undistort_image(image)
            debug_img = image.copy()
            drop_img = image.copy()

            edge_img = threshold(image, debug_img)
            wall_rect = detect_wall(edge_img, debug_img)
            wall_rect = order_wall_points(wall_rect)
            ellipses = detect_ellipses(edge_img, image, debug_img)
            wall_pose = detect_wall_pose(image, wall_rect, debug_img)

            ellipses_centers = []
            ellipses_centers_int = []
            for ellipse in ellipses:
                ellipses_centers.append([ellipse.center.x, ellipse.center.y])
                ellipses_centers_int.append([int(ellipse.center.x), int(ellipse.center.y)])

            if len(ellipses_centers) > 5:
                # An ellipse to rule them all
                ellipse_master = ransac(ellipses_centers)
                # cv.ellipse(debug_img, cv.fitEllipse(np.array(ellipses_centers_int)), (255, 0, 0), 2)
                if ellipse_master is not None:
                    cv.ellipse(debug_img, ellipse_master.raw, (0, 255, 255), 2)
                    plate_pose = detect_plate_pose(image, ellipse_master.center.raw, ellipses_centers, debug_img)
                    if plate_pose is not None and wall_pose is not None:
                        plate_3d_point = detect_plate_laser_point(image, plate_pose, debug_img)

                        wall_3d_points = get_3d_wall_points(wall_rect, plate_pose, wall_pose, debug_img)

                        # geometric_utility.get_plane(wall_3d_points[0], wall_3d_points[1], wall_3d_points[2])
                        top_3d_point, bottom_3d_point = detect_wall_line(image, wall_rect, wall_3d_points, debug_img)

                        new_camera_matrix, distortion = get_info_solvepnp()
                        aa, _ = cv.projectPoints(top_3d_point, plate_pose.r, plate_pose.t, new_camera_matrix, distortion)
                        bb, _ = cv.projectPoints(bottom_3d_point, plate_pose.r, plate_pose.t, new_camera_matrix, distortion)

                        cv.circle(debug_img, aa.astype('int')[0][0], 10, (0, 255, 0), cv.FILLED)
                        cv.circle(debug_img, bb.astype('int')[0][0], 10, (0, 255, 0), cv.FILLED)


            # cv.imwrite(".\\data\\debug\\image.jpg", debug_img)
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
