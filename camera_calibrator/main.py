import numpy as np
import cv2 as cv
import pickle
import shutil
import os

CHECKERBOARD = (9, 6)
NUM_OF_FRAMES = 250
DEBUG = False


def clear_debug_folder():
    shutil.rmtree('.\\data\\debug')
    os.makedirs('.\\data\\debug')


def main():
    if DEBUG:
        clear_debug_folder()

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    points_3d = []  # 3d point in real world space
    points_2d = []  # 2d points in image plane.
    images = []

    # Load the video
    video = cv.VideoCapture('.\\data\\calibration.mov')

    # Loop the frames in the video and take NUM_OF_FRAMES equally spaced frames
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    step = video_length // NUM_OF_FRAMES
    for i in range(NUM_OF_FRAMES):
        video.set(cv.CAP_PROP_POS_FRAMES, i*step)
        success, image = video.read()
        if success:
            images.append(image)

    # Loop and search for the chessboard in the selected frames
    print('Read %d frames' % len(images))
    useful_frames = 0
    for i, image in enumerate(images):
        # convert to grayscale
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray_img, CHECKERBOARD, None)
        # cv.CALIB_CB_ADAPTIVE_THRESH +
        # cv.CALIB_CB_FAST_CHECK +
        # cv.CALIB_CB_NORMALIZE_IMAGE)

        # Chessboard found
        if found:
            useful_frames += 1
            points_3d.append(objp)

            # Refining pixel coordinates for given 2d points
            corners_2d = cv.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            points_2d.append(corners_2d)

            # Draw and display the corners
            if DEBUG:
                image = cv.drawChessboardCorners(image, CHECKERBOARD, corners_2d, found)

        if DEBUG:
            cv.imwrite(".\\data\\debug\\frame%d.jpg" % (i*step), image)  # save frame as JPEG file

    print('Used %d frames' % useful_frames)

    if useful_frames > 0:
        # Perform camera calibration by
        # passing the value of above found out 3D points
        # and its corresponding pixel coordinates of the
        # detected corners
        ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(
            points_3d, points_2d, gray_img.shape[::-1], None, None)

        # Save the camera calibration result for later use
        pickle.dump((matrix, distortion), open(".\\data\\calibration.pkl", "wb"))

        # Displaying result
        print("Camera matrix:", matrix)
        print("\nDistortion coefficient:", distortion)
        # print("\nRotation Vectors:", r_vecs)
        # print("\nTranslation Vectors: ", t_vecs)

        print("RMS: ", ret)
    else:
        print("No useful frames")


if __name__ == '__main__':
    main()
