import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import camera_calibration as cc
import color_transforms as ct
import perspective_transforms as pt
import find_lanes as fl


class LaneFinder:

    left_fit = None
    right_fit = None
    is_initial = True

    def __init__(self):
        _, mtx, dist, _, _ = cc.get_calibration_coefficients('camera_cal', 9, 6)
        self.mtx = mtx
        self.dist = dist

    def find_lanes(self, img):
        img = cc.undistort_image(img, self.mtx, self.dist)
        combined_binary = self.color_and_gradient_transform(img)
        warped_img, M = pt.warp_image(combined_binary)

        if self.is_initial:
            self.left_fit, self.right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = fl.find_lane_pixels(
                warped_img)
            self.is_initial = False
        else:
            self.left_fit, self.right_fit = fl.search_around_polly(warped_img, self.left_fit, self.right_fit)

        return warped_img, self.left_fit, self.right_fit, M

    def color_and_gradient_transform(self, img):
        kernel_size = 3

        sobelx_binary = ct.abs_sobel_thresh(img, orient='x', sobel_kernel=kernel_size, thresh=(20, 100))
        sobely_binary = ct.abs_sobel_thresh(img, orient='y', sobel_kernel=kernel_size, thresh=(20, 100))
        mag_binary = ct.mag_thresh(img, sobel_kernel=kernel_size, thresh=(20, 100))
        dir_binary = ct.dir_threshold(img, sobel_kernel=kernel_size, thresh=(0.7, 1.4))
        hls_binary = ct.hls_select(img, thresh=(150, 255))

        combined_binary = ct.combine_color_transforms(sobelx_binary, sobely_binary, mag_binary, dir_binary, hls_binary)
        return combined_binary

    def lane_drawing_pipeline(self, img):
        warped_img, left_fit, right_fit, M = self.find_lanes(img)
        left_curvature, right_curvature, center = fl.get_curvature(warped_img, left_fit, right_fit)

        return fl.draw_lanes(img, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, center)


if __name__ == '__main__':
    images = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
    laneFinder = LaneFinder()
    for fname in images:
        img = mpimg.imread('test_images/' + fname)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = laneFinder.lane_drawing_pipeline(img)
        cv2.imshow('', img)
        cv2.waitKey(0)

    # laneFinder = LaneFinder()
    #
    # def process_image(img):
    #     return laneFinder.lane_drawing_pipeline(img)
    #
    #
    # from moviepy.editor import VideoFileClip
    #
    # white_output = 'output_video/project_video.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # white_clip = clip1.fl_image(process_image)
    # white_clip.write_videofile(white_output, audio=False)
