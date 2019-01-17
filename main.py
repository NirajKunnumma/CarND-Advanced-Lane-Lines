import os
import cv2
import matplotlib.image as mpimg

import camera_calibration as cc
import color_transforms as ct
import perspective_transforms as pt
import find_lanes as fl


class LaneFinder:

    left_fit = None
    right_fit = None
    prev_left = None
    prev_right = None
    count = 0
    reset = 0

    def __init__(self):
        _, mtx, dist, _, _ = cc.get_calibration_coefficients('camera_cal', 9, 6)
        self.mtx = mtx
        self.dist = dist

    def find_lanes(self, img):
        img = cc.undistort_image(img, self.mtx, self.dist)
        combined_binary = self.color_and_gradient_transform(img)
        warped_img, M = pt.warp_image(combined_binary)

        if self.count == 0:
            self.left_fit, self.right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = fl.find_lane_pixels(
                warped_img)
        else:
            self.left_fit, self.right_fit = fl.search_around_polly(warped_img, self.left_fit, self.right_fit)

        status = fl.validate_lines(self.left_fit, self.right_fit)

        if status == True:
            self.prev_left, self.prev_right = self.left_fit, self.right_fit
            self.count += 1
            self.reset = 0
        else:
            # Reset
            if self.reset > 4:
                self.left_fit, self.right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = fl.find_lane_pixels(
                    warped_img)
                self.reset = 0
            else:
                self.left_fit, self.right_fit = self.prev_left, self.prev_right

                self.reset += 1

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
    write_path = 'output_images/lane_lines/'

    laneFinder = LaneFinder()
    for image_file in os.listdir('test_images/'):
        if image_file.endswith('.jpg'):
            image = mpimg.imread(os.path.join('test_images/', image_file))
            cv2.imwrite(write_path + 'original_' + image_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image = laneFinder.lane_drawing_pipeline(image)
            cv2.imwrite(write_path + 'lanes_' + image_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


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
