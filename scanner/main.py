import random
from typing import List, Tuple
import point2ellipse
import polar_utility
from circular_marker import CircularMarker, MarkerColors
import numpy as np
import cv2 as cv
import imutils

from geometry import Rectangle
from undistorion import undistort_image, get_new_camera_matrix, get_distortion, get_h_w


def nothing(x):
    pass


def get_angle(center: Tuple[float, float], point: Tuple[float, float]):
    x = point[0] - center[0]
    y = point[1] - center[1]
    _, angle = polar_utility.cartesian_to_polar(x, y)
    return angle


def marker_positioning(image, center: Tuple[float, float], centers: List[List[int]]):
    circular_marker = CircularMarker()
    export = image.copy()

    img_points = []
    dst_points = []
    marker_idx_img_points = [[] for _ in circular_marker.points]
    centers = sorted(centers, reverse=True, key=lambda x: get_angle(center, x))

    i = 0
    while i < len(centers):
        export = cv.putText(export, str(i), centers[i], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img_points_temp = [
                centers[i],
                centers[(i + 1) % len(centers)],
                centers[(i + 2) % len(centers)],
                centers[(i + 3) % len(centers)],
             ]

        colors = []
        for m in img_points_temp:
            colors.append(MarkerColors.get_from_pixel(image, m[0], m[1]))

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

    export = imutils.resize(export, height=600)
    cv.imshow("Img_3", export)

    if len(img_points) > 4:
        #   ---------------- find homograpy
        M, mask = cv.findHomography(np.array(img_points).astype('float32'), np.array([[m[0]+300, m[1]+300] for m in dst_points]))
        img_out = cv.warpPerspective(image, M, (600, 600))

        img_out = imutils.resize(img_out, height=600)
        # cv.imshow("Img_3", img_out)

        # new_camera_matrix = get_new_camera_matrix()
        # distortion = np.zeros((4, 1))  # success get_distortion()
        # success, rotation_vector, translation_vector = cv.solvePnP(np.array(dst_points), np.array(img_points).astype('float32'), new_camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE)
        #
        # # test_point, _ = cv.projectPoints(np.array(circular_marker.get_marker_point(marker_idx-1)), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # # test_point_x = int(test_point[0][0][0])
        # # test_point_y = int(test_point[0][0][1])
        # # zero_point, _ = cv.projectPoints(np.array([0.0, 0.0, 0.0]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # # zero_point_x = int(zero_point[0][0][0])
        # # zero_point_y = int(zero_point[0][0][1])
        # # cv.line(export, (0, 0), (zero_point_x, zero_point_y), (0, 0, 0), 4)
        # # cv.imwrite(".\\data\\debug\\image.jpg", export)
        #
        # # if 0 <= test_point_x < w and 0 <= test_point_y < h and \
        # #    MarkerColors.get_from_pixel(image[test_point_y][test_point_x]) == circular_marker.get_marker_color(marker_idx-1):
        # zero, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # x_axis, jacobian = cv.projectPoints(np.array([(1000.0, 0.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # y_axis, _ = cv.projectPoints(np.array([(0.0, 1000.0, 0.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        # z_axis, _ = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, new_camera_matrix, distortion)
        #
        # return zero[0][0].astype('int'), x_axis[0][0].astype('int'), y_axis[0][0].astype('int'), z_axis[0][0].astype('int')

    return None
    # rotation_vector = cv.Rodrigues(rotation_vector)
    # translation_vector = cv.Rodrigues(translation_vector)


def ransac(centers: List[List[float]]):
    n = cv.getTrackbarPos('n', 'Ransac')
    k = cv.getTrackbarPos('k', 'Ransac')
    t = cv.getTrackbarPos('t', 'Ransac')
    d = cv.getTrackbarPos('d', 'Ransac')

    n = max(n, 5)

    best_model = None
    best_consensus_set = None
    best_error = 0
    for iteration in range(k):
        possible_inliers_idx = set(random.sample([i for i in range(len(centers))], n))
        possible_inliers = [centers[i] for i in possible_inliers_idx]
        possible_model = cv.fitEllipse(np.array(possible_inliers))
        consensus_set_idx = possible_inliers_idx

        for i, c in enumerate(centers):
            if i not in possible_inliers_idx:
                dist = point2ellipse.point_ellipse_distance(possible_model, (c[0], c[1]))
                if dist < t:
                    consensus_set_idx.add(i)

        if len(consensus_set_idx) >= d:
            enhanced_possible_inliers = [centers[i] for i in consensus_set_idx]
            enhanced_model = cv.fitEllipse(np.array(enhanced_possible_inliers))
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
    cv.createTrackbar('t', 'Ransac', 25, 40, nothing)
    cv.createTrackbar('d', 'Ransac', 7, 30, nothing)

    video = cv.VideoCapture('.\\data\\ball.mov')

    # Loop the frames in the video and take NUM_OF_FRAMES equally spaced frames
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_i in range(video_length):
        success, image = video.read()
        if success:
            image = undistort_image(image)
            clone = image.copy()
            gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            threshold = cv.getTrackbarPos('threshold', 'Parameters')
            ellipses_precision = cv.getTrackbarPos('ellipses_precision', 'Parameters')
            ellipses_min_points = cv.getTrackbarPos('ellipses_min_points', 'Parameters')
            ellipses_max_points = cv.getTrackbarPos('ellipses_max_points', 'Parameters')
            ellipses_ratio = cv.getTrackbarPos('ellipses_ratio', 'Parameters')
            sigma = cv.getTrackbarPos('sigma', 'Parameters')
            a_canny = cv.getTrackbarPos('a_canny', 'Parameters')
            b_canny = cv.getTrackbarPos('b_canny', 'Parameters')

            ret, thresh_img = cv.threshold(gray_img, threshold, 255, 0)

            # Blur an image
            bilateral_filtered_image = cv.bilateralFilter(thresh_img, 5, sigma, sigma)

            morph_size = 2
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))

            # Detect edges
            edge_detected_image = cv.Canny(bilateral_filtered_image, a_canny, b_canny)

            edge_detected_image = cv.morphologyEx(edge_detected_image, cv.MORPH_CLOSE, kernel, iterations=4)

            # Find contours
            contours, hierarchy = cv.findContours(edge_detected_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # Find the rotated rectangles and ellipses for each contour
            rects: List[Rectangle] = []
            ellipses: List[cv.typing.RotatedRect | None] = [None] * len(contours)

            for i, c in enumerate(contours):
                # x1, y1 = c[0][0]
                points = cv.approxPolyDP(c, 0.01*cv.arcLength(c, True), True)
                first_child = hierarchy[0][i][2]
                if len(points) == 4 and first_child == -1:
                    rects.append(Rectangle(cv.boundingRect(c), points))

                if ellipses_min_points < c.shape[0] < ellipses_max_points:
                    # rects[i] = cv.minAreaRect(c)
                    ellipses[i] = cv.fitEllipse(c)

            wall_rect = max(rects, key=lambda x: x.area)
            for p in range(len(wall_rect.points)):
                clone = cv.line(clone, wall_rect.points[p][0], wall_rect.points[(p + 1) % len(wall_rect.points)][0], (0, 0, 255), 2)

            for i, c in enumerate(contours):
                if ellipses[i] is not None:
                    max_size = max(ellipses[i][1][0], ellipses[i][1][1])
                    min_size = min(ellipses[i][1][0], ellipses[i][1][1])
                    if max_size > min_size * (ellipses_ratio / 100):
                        ellipses[i] = None
                    else:
                        mean_err = 0
                        max_err = 0
                        for p in c:
                            err = point2ellipse.point_ellipse_distance(ellipses[i], (p[0][0], p[0][1]))
                            mean_err += err / len(c)
                        if mean_err > ellipses_precision:
                            ellipses[i] = None

            for i, e in enumerate(ellipses):
                if e is not None:
                    center = e[0]
                    size = e[1][0] * e[1][1]

                    for i2, e2 in enumerate(ellipses):
                        if e2 is not None and i != i2:
                            center2 = e2[0]
                            if np.sqrt(((center[0] - center2[0])**2) + ((center[1] - center2[1])**2)) < max(e2[1][0], e2[1][1]):
                                size2 = e2[1][0] * e2[1][1]
                                if size2 > size:
                                    pass
                                    ellipses[i] = None

            color = (0, 255, 0)
            centers = []
            for ellipse in ellipses:
                if ellipse is not None:
                    centers.append([int(ellipse[0][0]), int(ellipse[0][1])])
                    # contour
                    # cv.drawContours(drawing, contours, i, color)
                    # ellipse
                    cv.ellipse(clone, ellipse, MarkerColors.get_from_pixel_debug(image, int(ellipse[0][0]), int(ellipse[0][1])), 3)
                    # rotated rectangle
                    # box = cv.boxPoints(rects[i])
                    # box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                    # cv.drawContours(drawing, [box], 0, color)

            if len(centers) > 5:
                gg = ransac(centers)
                # cv.ellipse(clone, cv.fitEllipse(np.array(centers)), (255, 0, 0), 2)
                if gg is not None:
                    # cv.ellipse(clone, gg, (0, 255, 255), 2)
                    jk = marker_positioning(image, gg[0], centers)
                    if jk is not None:
                        zero, x_axis, y_axis, z_axis = jk
                        cv.line(clone, zero, x_axis, (0, 0, 0), 2)
                        cv.line(clone, zero, y_axis, (0, 0, 0), 2)
                        cv.line(clone, zero, z_axis, (0, 0, 0), 2)

            # cv.imwrite(".\\data\\debug\\image.jpg", clone)
            clone = imutils.resize(clone, height=600)
            thresh_img = imutils.resize(thresh_img, height=600)
            edge_detected_image = imutils.resize(edge_detected_image, height=600)
            cv.imshow("Img_1", clone)
            cv.imshow("Img_2", edge_detected_image)

            # ESC to break
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break


if __name__ == '__main__':
    circular_marker = CircularMarker()

    a = circular_marker.get_markers_position([MarkerColors.Black, MarkerColors.Magenta, MarkerColors.Magenta, MarkerColors.Cyan])
    b = circular_marker.get_markers_position([MarkerColors.Cyan, MarkerColors.Yellow, MarkerColors.White, MarkerColors.Magenta])
    c = circular_marker.get_markers_position([MarkerColors.Cyan, MarkerColors.Magenta, MarkerColors.Magenta, MarkerColors.Magenta])

    main()
    cv.destroyAllWindows()
    pass
