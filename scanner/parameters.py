import cv2 as cv


def nothing(x):
    pass


ENABLE_PARAMETERS = False
ENABLE_RANSAC = False
ENABLE_RED_FILTER = False
ENABLE_RED_FILTER2 = False


def init_parameters():
    if ENABLE_PARAMETERS:
        cv.namedWindow("Parameters")
        cv.createTrackbar('threshold', 'Parameters', Parameters.threshold, 255, nothing)
        cv.createTrackbar('ellipses_precision', 'Parameters', Parameters.ellipses_precision, 100, nothing)
        cv.createTrackbar('ellipses_min_points', 'Parameters', Parameters.ellipses_min_points, 100, nothing)
        cv.createTrackbar('ellipses_max_points', 'Parameters', Parameters.ellipses_max_points, 1000, nothing)
        cv.createTrackbar('ellipses_ratio', 'Parameters', Parameters.ellipses_ratio, 1000, nothing)
        cv.createTrackbar('sigma', 'Parameters', Parameters.sigma, 255, nothing)
        cv.createTrackbar('a_canny', 'Parameters', Parameters.a_canny, 255, nothing)
        cv.createTrackbar('b_canny', 'Parameters', Parameters.b_canny, 255, nothing)

    if ENABLE_RANSAC:
        cv.namedWindow("Ransac")
        cv.createTrackbar('n', 'Ransac', Parameters.ransac_n, 30, nothing)
        cv.createTrackbar('k', 'Ransac', Parameters.ransac_k, 500, nothing)
        cv.createTrackbar('t', 'Ransac', Parameters.ransac_t, 400, nothing)
        cv.createTrackbar('d', 'Ransac', Parameters.ransac_d, 30, nothing)

    if ENABLE_RED_FILTER:
        cv.namedWindow("Red_Filter")
        cv.createTrackbar('b_dw', 'Red_Filter', Parameters.laser_b_dw, 255, nothing)
        cv.createTrackbar('b_up', 'Red_Filter', Parameters.laser_b_up, 255, nothing)
        cv.createTrackbar('g_dw', 'Red_Filter', Parameters.laser_g_dw, 255, nothing)
        cv.createTrackbar('g_up', 'Red_Filter', Parameters.laser_g_up, 255, nothing)
        cv.createTrackbar('r_dw', 'Red_Filter', Parameters.laser_r_dw, 255, nothing)
        cv.createTrackbar('r_up', 'Red_Filter', Parameters.laser_r_up, 255, nothing)

    if ENABLE_RED_FILTER2:
        cv.namedWindow("Red_Filter2")
        cv.createTrackbar('h_dw', 'Red_Filter2', Parameters.laser_h_dw, 360//2, nothing)
        cv.createTrackbar('h_up', 'Red_Filter2', Parameters.laser_h_up, 360//2, nothing)
        cv.createTrackbar('s_dw', 'Red_Filter2', Parameters.laser_s_dw, 255, nothing)
        cv.createTrackbar('s_up', 'Red_Filter2', Parameters.laser_s_up, 255, nothing)
        cv.createTrackbar('v_dw', 'Red_Filter2', Parameters.laser_v_dw, 255, nothing)
        cv.createTrackbar('v_up', 'Red_Filter2', Parameters.laser_v_up, 255, nothing)


def update_parameters():
    if ENABLE_PARAMETERS:
        Parameters.threshold = cv.getTrackbarPos('threshold', 'Parameters')
        Parameters.ellipses_precision = cv.getTrackbarPos('ellipses_precision', 'Parameters')
        Parameters.ellipses_min_points = cv.getTrackbarPos('ellipses_min_points', 'Parameters')
        Parameters.ellipses_max_points = cv.getTrackbarPos('ellipses_max_points', 'Parameters')
        Parameters.ellipses_ratio = cv.getTrackbarPos('ellipses_ratio', 'Parameters')
        Parameters.sigma = cv.getTrackbarPos('sigma', 'Parameters')
        Parameters.a_canny = cv.getTrackbarPos('a_canny', 'Parameters')
        Parameters.b_canny = cv.getTrackbarPos('b_canny', 'Parameters')

    if ENABLE_RANSAC:
        Parameters.ransac_n = cv.getTrackbarPos('n', 'Ransac')
        Parameters.ransac_k = cv.getTrackbarPos('k', 'Ransac')
        Parameters.ransac_t = cv.getTrackbarPos('t', 'Ransac')
        Parameters.ransac_d = cv.getTrackbarPos('d', 'Ransac')

    if ENABLE_RED_FILTER:
        Parameters.laser_b_dw = cv.getTrackbarPos('b_dw', 'Red_Filter')
        Parameters.laser_b_up = cv.getTrackbarPos('b_up', 'Red_Filter')
        Parameters.laser_g_dw = cv.getTrackbarPos('g_dw', 'Red_Filter')
        Parameters.laser_g_up = cv.getTrackbarPos('g_up', 'Red_Filter')
        Parameters.laser_r_dw = cv.getTrackbarPos('r_dw', 'Red_Filter')
        Parameters.laser_r_up = cv.getTrackbarPos('r_up', 'Red_Filter')

    if ENABLE_RED_FILTER2:
        Parameters.laser_h_dw = cv.getTrackbarPos('h_dw', 'Red_Filter2')
        Parameters.laser_h_up = cv.getTrackbarPos('h_up', 'Red_Filter2')
        Parameters.laser_s_dw = cv.getTrackbarPos('s_dw', 'Red_Filter2')
        Parameters.laser_s_up = cv.getTrackbarPos('s_up', 'Red_Filter2')
        Parameters.laser_v_dw = cv.getTrackbarPos('v_dw', 'Red_Filter2')
        Parameters.laser_v_up = cv.getTrackbarPos('v_up', 'Red_Filter2')


class Parameters:
    threshold: int = 87
    ellipses_precision: int = 30
    ellipses_min_points: int = 30
    ellipses_max_points: int = 300
    ellipses_ratio: int = 200
    sigma: int = 150
    a_canny: int = 75
    b_canny: int = 200

    # ransac
    ransac_n: int = 5
    ransac_k: int = 35
    ransac_t: int = 400  # 25
    ransac_d: int = 7

    # red filter
    laser_b_dw: int = 150
    laser_b_up: int = 195
    laser_g_dw: int = 150
    laser_g_up: int = 195
    laser_r_dw: int = 205
    laser_r_up: int = 255

    # red filter2
    laser_h_dw: int = 157
    laser_h_up: int = 180
    laser_s_dw: int = 50
    laser_s_up: int = 165
    laser_v_dw: int = 190
    laser_v_up: int = 255
