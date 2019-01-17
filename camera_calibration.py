import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_calibration_coefficients(image_path, nx, ny):
    images = [mpimg.imread(os.path.join(image_path, image_file)) for image_file in os.listdir(image_path) if
              image_file.endswith('.jpg')]

    object_points = []
    image_points = []

    object_point = np.zeros((nx * ny, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            image_points.append(corners)
            object_points.append(object_point)

    gray_image_shape = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_image_shape, None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort_image(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = get_calibration_coefficients('camera_cal', 9, 6)

    for image_file in os.listdir('camera_cal/'):
        if image_file.endswith('.jpg'):
            image = mpimg.imread(os.path.join('camera_cal/', image_file))
            undst = cv2.undistort(image, mtx, dist, None, mtx)
            cv2.imwrite(os.path.join('output_images/undistorted_images/chessboard/', image_file),
                        cv2.cvtColor(undst, cv2.COLOR_RGB2BGR))

    for image_file in os.listdir('test_images/'):
        if image_file.endswith('.jpg'):
            image = mpimg.imread(os.path.join('test_images/', image_file))
            undst = cv2.undistort(image, mtx, dist, None, mtx)
            cv2.imwrite(os.path.join('output_images/undistorted_images/lane_images/', image_file),
                        cv2.cvtColor(undst, cv2.COLOR_RGB2BGR))
