import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) if orient == 'y' else cv2.Sobel(gray, cv2.CV_64F, 1, 0,
                                                                                                 ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output


def combine_color_transforms(sobelx_binary, sobely_binary, mag_binary, dir_binary, hls_binary):
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[
        ((sobelx_binary == 1) & (sobely_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | hls_binary == 1] = 1

    return combined_binary


if __name__ == '__main__':
    # image = mpimg.imread('test_images/test2.jpg')
    #
    # temp_image = np.copy(image)
    #
    # kernel_size = 3
    #
    # sobelx_binary = abs_sobel_thresh(temp_image, orient='x', sobel_kernel=kernel_size, thresh=(20, 100))
    # sobely_binary = abs_sobel_thresh(temp_image, orient='y', sobel_kernel=kernel_size, thresh=(20, 100))
    # mag_binary = mag_thresh(temp_image, sobel_kernel=kernel_size, thresh=(20, 100))
    # dir_binary = dir_threshold(temp_image, sobel_kernel=kernel_size, thresh=(0.7, 1.4))
    # hls_binary = hls_select(temp_image, thresh=(150, 255))
    #
    # combined_binary = combine_color_transforms(sobelx_binary, sobely_binary, mag_binary, dir_binary, hls_binary)
    #
    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    # ax1.set_title('Actual image')
    # ax1.imshow(image)
    # ax2.set_title('Color thresholding')
    # ax2.imshow(hls_binary, cmap='gray')
    # ax3.set_title('Combined all')
    # ax3.imshow(combined_binary, cmap='gray')
    # plt.show()

    write_path = 'output_images/color_and_gradient_thresholds/'

    pt_write_path = 'output_images/perspective_transforms/'

    import perspective_transforms as pt

    for image_file in os.listdir('test_images/'):
        if image_file.endswith('.jpg'):
            image = mpimg.imread(os.path.join('test_images/', image_file))

            kernel_size = 3

            sobelx_binary = abs_sobel_thresh(image, orient='x', sobel_kernel=kernel_size, thresh=(20, 100))
            cv2.imwrite(write_path + 'sobelx_' + image_file, sobelx_binary * 255)

            sobely_binary = abs_sobel_thresh(image, orient='y', sobel_kernel=kernel_size, thresh=(20, 100))
            cv2.imwrite(write_path + 'sobely_' + image_file, sobely_binary * 255)

            mag_binary = mag_thresh(image, sobel_kernel=kernel_size, thresh=(20, 100))
            cv2.imwrite(write_path + 'mag_binary_' + image_file, mag_binary * 255)

            dir_binary = dir_threshold(image, sobel_kernel=kernel_size, thresh=(0.7, 1.4))
            cv2.imwrite(write_path + 'dir_binary_' + image_file, dir_binary * 255)

            hls_binary = hls_select(image, thresh=(150, 255))
            cv2.imwrite(write_path + 'hls_binary_' + image_file, hls_binary * 255)

            combined_binary = combine_color_transforms(sobelx_binary, sobely_binary, mag_binary, dir_binary, hls_binary)
            cv2.imwrite(write_path + 'combined_binary_' + image_file, combined_binary * 255)
            cv2.imwrite(pt_write_path + 'original_binary_' + image_file, combined_binary * 255)

            transformed_image, _ = pt.warp_image(combined_binary * 255)
            cv2.imwrite(pt_write_path + 'warped_' + image_file, transformed_image)

