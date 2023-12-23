import pickle
import cv2 as cv

_matrix = None
_distortion = None
_new_camera_matrix = None
_roi = None


def undistort_image(image):
    global _matrix
    global _distortion
    global _new_camera_matrix
    global _roi

    if _matrix is None or _distortion is None:
        (_matrix, _distortion) = pickle.load(open(".\\data\\calibration.pkl", "rb"))

    if _new_camera_matrix is None or _roi is None:
        h, w = image.shape[:2]
        _new_camera_matrix, _roi = cv.getOptimalNewCameraMatrix(_matrix, _distortion, (w, h), 1, (w, h))

    dst = cv.undistort(image, _matrix, _distortion, None, _new_camera_matrix)
    # crop the image
    x, y, w, h = _roi
    dst = dst[y:y + h, x:x + w]
    return dst
