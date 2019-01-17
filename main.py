import os
import cv2
import argparse
import matplotlib.image as mpimg

import camera_calibration as cc
import color_transforms as ct
import perspective_transforms as pt
import find_lanes as fl
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser(description="Advanced Lane Finding")


class LaneFinder:
    left_fit = None
    right_fit = None
    is_initial = True

    def __init__(self, mtx, dist):

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


def initialize_parser():
    parser.add_argument("-i", "--input", required=True, help="Path to input files")
    parser.add_argument("-o", "--output", required=True, help="Path to output files")


if __name__ == '__main__':

    initialize_parser()
    args = parser.parse_args()

    print("1. Starting Camera Calibration")
    _, mtx, dist, _, _ = cc.get_calibration_coefficients('camera_cal', 9, 6)
    print("2. Camera Calibrations Obtained")

    print("3. Finding Lanes....")
    if args.input and len(args.input) and args.output and len(args.output):
        input_path = args.input
        output_path = args.output
        for input_file in os.listdir(input_path):
            if input_file.endswith('.jpg'):
                laneFinder = LaneFinder(mtx, dist)
                image = mpimg.imread(os.path.join(input_path, input_file))
                image = laneFinder.lane_drawing_pipeline(image)
                cv2.imwrite(os.path.join(output_path, input_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            elif input_file.endswith('.mp4'):
                laneFinder = LaneFinder(mtx, dist)

                def process_image(img):
                    return laneFinder.lane_drawing_pipeline(img)

                white_output = os.path.join(output_path, input_file)
                clip1 = VideoFileClip(os.path.join(input_path, input_file))
                white_clip = clip1.fl_image(process_image)
                white_clip.write_videofile(white_output, audio=False)
    else:
        print("Please specify input and output folders")


