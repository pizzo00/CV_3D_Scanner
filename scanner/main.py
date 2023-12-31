import random
from typing import List, Tuple
import point2ellipse
import polar_utility
from circular_marker import CircularMarker, MarkerColors
import numpy as np
import cv2 as cv
import imutils

from geometry import Rectangle, Ellipse
from undistorion import undistort_image, get_new_camera_matrix, get_distortion, get_h_w


def nothing(x):
    pass


def invert_pose(r, t):
    #np_rodrigues = np.asarray(r[:, :], np.float64)
    rmat = cv.Rodrigues(r)[0]
    return np.matrix(rmat).T, (-np.matrix(rmat).T) @ np.matrix(t)


def get_info_solvepnp():
    new_camera_matrix = get_new_camera_matrix()
    distortion = np.zeros((4, 1))  # success get_distortion()
    return new_camera_matrix, distortion


def marker_positioning(image, center: Tuple[float, float], centers: List[List[float]], debug_img):
    circular_marker = CircularMarker()
    export = image.copy()

    img_points = []
    dst_points = []
    marker_idx_img_points = [[] for _ in circular_marker.points]
    centers = sorted(centers, reverse=True, key=lambda x: polar_utility.get_angle(center, x))

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

        # for m in markers:
        #     cv.line(export, (0, 0), m, MarkerColors.get_from_pixel_debug(image, m[0], m[1]), 2)
        # cv.imwrite(".\\data\\debug\\image.jpg", export)

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
        success, rotation_vector, translation_vector = cv.solvePnP(np.array(dst_points), np.array(img_points), new_camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
        ri, ti = invert_pose(rotation_vector, translation_vector)

        if not success:
            return None, None, debug_img
        # test_point, _ = cv.projectPoints(np.array(circular_marker.get_marker_point(marker_idx-1)), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # test_point_x = int(test_point[0][0][0])
        # test_point_y = int(test_point[0][0][1])
        # zero_point, _ = cv.projectPoints(np.array([0.0, 0.0, 0.0]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # zero_point_x = int(zero_point[0][0][0])
        # zero_point_y = int(zero_point[0][0][1])
        # cv.line(export, (0, 0), (zero_point_x, zero_point_y), (0, 0, 0), 4)
        # cv.imwrite(".\\data\\debug\\image.jpg", export)

        # if 0 <= test_point_x < w and 0 <= test_point_y < h and \
        #    MarkerColors.get_from_pixel(image[test_point_y][test_point_x]) == circular_marker.get_marker_color(marker_idx-1):
        zero, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        x_axis, jacobian = cv.projectPoints(np.array([(100.0, 0.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        y_axis, _ = cv.projectPoints(np.array([(0.0, 100.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        z_axis, _ = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        pose, _ = cv.projectPoints(np.array([ti.item(0), ti.item(1), ti.item(2)]), rotation_vector, translation_vector, new_camera_matrix, distortion)

        cv.line(debug_img, zero[0][0].astype('int'), x_axis[0][0].astype('int'), (255, 0, 0), 2)
        cv.line(debug_img, zero[0][0].astype('int'), y_axis[0][0].astype('int'), (0, 255, 0), 2)
        cv.line(debug_img, zero[0][0].astype('int'), z_axis[0][0].astype('int'), (0, 0, 255), 2)
        cv.line(debug_img, zero[0][0].astype('int'), pose[0][0].astype('int'), (0, 0, 255), 10)

        return rotation_vector, translation_vector, debug_img

    return None, None, debug_img
    # rotation_vector = cv.Rodrigues(rotation_vector)
    # translation_vector = cv.Rodrigues(translation_vector)


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

    return edge_img, debug_img


MORPH_SIZE = 2


def detect_wall(edge_img, debug_img) -> Tuple[Rectangle, cv.typing.MatLike]:
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

    return wall_rect, debug_img


def detect_ellipses(edge_img, image, debug_img) -> Tuple[List[Ellipse | None], cv.typing.MatLike]:
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

    return ellipses, debug_img


def detect_wall_line(image, wall_rect: Rectangle, debug_img):
    h = 230
    w = 130
    h_dec = 230.0
    w_dec = 130.0
    margin = 5  # Prevent black pixel

    unpacked_points = [i[0] * 1.0 for i in wall_rect.points]
    points_y_sorted = sorted(unpacked_points, key=lambda x: x[1])
    top_points = sorted(points_y_sorted[:2], key=lambda x: x[0])
    bottom_points = sorted(points_y_sorted[2:], key=lambda x: x[0])

    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    img_points = top_points + bottom_points
    # world zero on top left
    dst_points = [
        [0.0, 0.0, 0.0],
        [w_dec, 0.0, 0.0],
        [0.0, h_dec, 0.0],
        [w_dec, h_dec, 0.0],
    ]

    M, mask = cv.findHomography(np.array(img_points), np.array(dst_points))
    homo_img = cv.warpPerspective(image, M, (w, h))

    lower_red = np.array([170, 170, 205], dtype="uint8")
    upper_red = np.array([255, 255, 255], dtype="uint8")
    mask = cv.inRange(homo_img, lower_red, upper_red)
    edge_img = cv.bitwise_and(homo_img, homo_img, mask=mask)

    points_of_line = []
    for x in range(w):
        if margin < x < w - margin:
            for y in range(h):
                if margin < y < h - margin:
                    if mask[y][x] == 255:
                        points_of_line.append([[x, y]])

    if len(points_of_line) < 5:
        return None, None, debug_img
    line = cv.fitLine(np.array(points_of_line), cv.DIST_L2, 0, 0.01, 0.01)
    vx = line[0][0]
    vy = line[1][0]
    x0 = line[2][0]
    y0 = line[3][0]
    m = 200
    cv.line(homo_img, np.array([int(x0 - m * vx), int(y0 - m * vy)]), np.array([int(x0 + m * vx), int(y0 + m * vy)]), (255, 255, 0))

    homo_img = imutils.resize(homo_img, height=600)
    cv.imshow("Img_3", homo_img)

    new_camera_matrix, distortion = get_info_solvepnp()
    success, rotation_vector, translation_vector = cv.solvePnP(np.array(dst_points), np.array(img_points), new_camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
    ri, ti = invert_pose(rotation_vector, translation_vector)

    if not success:
        return None, None, debug_img

    zero, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
    x_axis, jacobian = cv.projectPoints(np.array([(100.0, 0.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
    y_axis, _ = cv.projectPoints(np.array([(0.0, 100.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
    z_axis, _ = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
    pose, _ = cv.projectPoints(ti, rotation_vector, translation_vector, new_camera_matrix, distortion)

    cv.line(debug_img, zero[0][0].astype('int'), x_axis[0][0].astype('int'), (255, 0, 0), 2)
    cv.line(debug_img, zero[0][0].astype('int'), y_axis[0][0].astype('int'), (0, 255, 0), 2)
    cv.line(debug_img, zero[0][0].astype('int'), z_axis[0][0].astype('int'), (0, 0, 255), 2)
    cv.line(debug_img, zero[0][0].astype('int'), pose[0][0].astype('int'), (0, 0, 255), 10)

    return rotation_vector, translation_vector, debug_img


def main():
    cv.namedWindow("Img_1")
    cv.namedWindow("Img_2")
    cv.namedWindow("Img_3")
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
    cv.createTrackbar('t', 'Ransac', 25, 400, nothing)
    cv.createTrackbar('d', 'Ransac', 7, 30, nothing)

    video = cv.VideoCapture('.\\data\\cube.mov')

    # Loop the frames in the video and take NUM_OF_FRAMES equally spaced frames
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_i in range(video_length):
        success, image = video.read()
        if success:
            image = undistort_image(image)
            debug_img = image.copy()

            edge_img, debug_img = threshold(image, debug_img)
            wall_rect, debug_img = detect_wall(edge_img, debug_img)
            ellipses, debug_img = detect_ellipses(edge_img, image, debug_img)
            wall_r, wall_t, debug_img = detect_wall_line(image, wall_rect, debug_img)
            if wall_t is not None:
                wall_ri, wall_ti = invert_pose(wall_r, wall_t)
                distance_from_wall = np.sqrt(wall_ti.item(0) ** 2 + wall_ti.item(1) ** 2 + wall_ti.item(2) ** 2)
                debug_img = cv.putText(debug_img, str(int(distance_from_wall)), [50, 50], cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)


            ellipses_centers = []
            ellipses_centers_int = []
            for ellipse in ellipses:
                ellipses_centers.append([ellipse.center.x, ellipse.center.y])
                ellipses_centers_int.append([int(ellipse.center.x), int(ellipse.center.y)])

            if len(ellipses_centers) > 5:
                # An ellipse to rule them all
                ellipse_master = ransac(ellipses_centers)
                cv.ellipse(debug_img, cv.fitEllipse(np.array(ellipses_centers_int)), (255, 0, 0), 2)
                if ellipse_master is not None:
                    # cv.ellipse(debug_img, gg, (0, 255, 255), 2)
                    plate_r, plate_t, debug_img = marker_positioning(image, ellipse_master.center.raw, ellipses_centers, debug_img)

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
