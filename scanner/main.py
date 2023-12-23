import point2ellipse
from circular_marker import CircularMarker, MarkerColors
import numpy as np
import cv2 as cv
import imutils

from undistorion import undistort_image


def nothing(x):
    pass


def main():
    cv.namedWindow("Debug")
    cv.createTrackbar('threshold', 'Debug', 87, 255, nothing)
    cv.createTrackbar('ellipses_precision', 'Debug', 300, 300, nothing)
    cv.createTrackbar('ellipses_min_points', 'Debug', 30, 100, nothing)
    cv.createTrackbar('ellipses_max_points', 'Debug', 300, 1000, nothing)
    cv.createTrackbar('ellipses_ratio', 'Debug', 200, 1000, nothing)

    video = cv.VideoCapture('.\\data\\ball.mov')

    # Loop the frames in the video and take NUM_OF_FRAMES equally spaced frames
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_i in range(video_length):
        success, image = video.read()
        if success:
            image = undistort_image(image)
            clone = image.copy()
            gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            threshold = cv.getTrackbarPos('threshold', 'Debug')
            ellipses_precision = cv.getTrackbarPos('ellipses_precision', 'Debug')
            ellipses_min_points = cv.getTrackbarPos('ellipses_min_points', 'Debug')
            ellipses_max_points = cv.getTrackbarPos('ellipses_max_points', 'Debug')
            ellipses_ratio = cv.getTrackbarPos('ellipses_ratio', 'Debug')

            ret, thresh_img = cv.threshold(gray_img, threshold, 255, 0)

            # Blur an image
            bilateral_filtered_image = cv.bilateralFilter(thresh_img, 5, 175, 175)

            # Detect edges
            edge_detected_image = cv.Canny(bilateral_filtered_image, 75, 200)

            # Find contours
            contours, _ = cv.findContours(edge_detected_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # Find the rotated rectangles and ellipses for each contour
            rects = [None] * len(contours)
            ellipses = [None] * len(contours)
            for i, c in enumerate(contours):
                if ellipses_min_points < c.shape[0] < ellipses_max_points:
                    rects[i] = cv.minAreaRect(c)
                    ellipses[i] = cv.fitEllipse(c)
            # Draw contours + rotated rects + ellipses

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
                            err = point2ellipse.pointEllipseDistance(ellipses[i], (p[0][0], p[0][1]))
                            mean_err += err / len(c)
                        if mean_err > ellipses_precision:
                            ellipses[i] = None

            color = (0, 255, 0)
            centers = []
            for i, c in enumerate(contours):
                if ellipses[i] is not None:
                    centers.append([int(ellipses[i][0][0]), int(ellipses[i][0][1])])
                    # contour
                    # cv.drawContours(drawing, contours, i, color)
                    # ellipse
                    cv.ellipse(clone, ellipses[i], (0, 255, 0), 3)
                    # rotated rectangle
                    # box = cv.boxPoints(rects[i])
                    # box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                    # cv.drawContours(drawing, [box], 0, color)

            if len(centers) > 5:
                cv.ellipse(clone, cv.fitEllipse(np.array(centers)), (255, 0, 0), 2)

            clone = imutils.resize(clone, height=600)
            cv.imshow("Debug", clone)

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
