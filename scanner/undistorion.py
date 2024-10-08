import pickle
import cv2 as cv

_matrix = None
_distortion = None
_camera_matrix = None
_roi = None
_h = None
_w = None


def get_h_w():
    return _h, _w


def get_camera_matrix():
    return _camera_matrix


def get_distortion():
    return _distortion


def undistort_image(image):
    global _matrix
    global _distortion
    global _camera_matrix
    global _roi
    global _h
    global _w

    # Load output of calibrator
    if _matrix is None or _distortion is None:
        (_matrix, _distortion) = pickle.load(open(".\\data\\calibration.pkl", "rb"))

    # Create camera matrix
    if _camera_matrix is None or _roi is None or _h is None or _w is None:
        _h, _w = image.shape[:2]
        _camera_matrix, _roi = cv.getOptimalNewCameraMatrix(_matrix, _distortion, (_w, _h), 1, (_w, _h))

    # Undistort the image
    dst = cv.undistort(image, _matrix, _distortion, None, _camera_matrix)

    # Crop the image
    x, y, w, h = _roi
    dst = dst[y:y + h, x:x + w]
    return dst
