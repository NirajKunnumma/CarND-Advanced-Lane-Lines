import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_calibration_coefficients(image_path, nx, ny):
    images = [mpimg.imread(os.path.join(image_path, image_file)) for image_file in os.listdir(image_path) if
              image_file.endswith('.jpg')]

    print(len(images))

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

    print(gray_image_shape)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_image_shape, None, None)

    return ret, mtx, dist, rvecs, tvecs


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = get_calibration_coefficients('camera_cal', 9, 6)

    test_img = mpimg.imread('camera_cal/calibration2.jpg')

    undst = cv2.undistort(test_img, mtx, dist, None, mtx)

    plt.imshow(undst)
    plt.show()