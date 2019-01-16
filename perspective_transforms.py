import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def warp_image(gray_img):
    image = np.copy(gray_img)

    # Select point by viewing on image
    top_left = [575, 475]
    top_right = [725, 475]
    bottom_left = [275, 675]
    bottom_right = [1050, 675]

    src = np.float32([top_left, bottom_left, top_right, bottom_right])

    dst = np.float32([[200, 0], [200, 700], [1000, 0], [1000, 700]])

    # Grab the image shape
    img_size = (gray_img.shape[1], gray_img.shape[0])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)

    # Return the resulting image and matrix
    return warped, M


if __name__ == '__main__':
    image = mpimg.imread('test_images/test2.jpg')

    temp_image = np.copy(image)



    warped_image, M = warp_image(temp_image)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Actual image')
    ax1.imshow(image)
    ax2.set_title('Warped Image')
    ax2.imshow(warped_image)
    plt.show()

