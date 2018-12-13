import os
import cv2
import numpy as np

class GridTools:

    def __init__(self, is_debug=False, show_intermid=False):

        self.is_debug = is_debug
        self.show_intermid = show_intermid

        if is_debug:
            self.debug_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Debug', 'GridTools')

    def remove_grid(self, img, name=None):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bin_img = cv2.bitwise_not(gray)

        _, bin_thresh = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if self.is_debug and self.show_intermid:
            cv2.imwrite(os.path.join(self.debug_path, 'bin_thresh_{}.png'.format(name)), bin_thresh)

        kernel_1x2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        kernel_2x1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

        img_temp1 = cv2.erode(bin_thresh, ver_kernel, iterations=3)
        ver_lines_img = cv2.dilate(img_temp1, ver_kernel, iterations=3)

        if self.is_debug and self.show_intermid:
            cv2.imwrite(os.path.join(self.debug_path, 'ver_lines_img_{}.png'.format(name)), ver_lines_img)

        img_temp2 = cv2.erode(bin_thresh, hor_kernel, iterations=3)
        hor_lines_img = cv2.dilate(img_temp2, hor_kernel, iterations=3)

        if self.is_debug and self.show_intermid:
            cv2.imwrite(os.path.join(self.debug_path, 'hor_lines_img_{}.png'.format(name)), hor_lines_img)

        merged_lines = cv2.addWeighted(ver_lines_img, 0.5, hor_lines_img, 0.5, 0.0)

        _, grid_final_bin = cv2.threshold(merged_lines, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        grid_final_bin = cv2.dilate(grid_final_bin, kernel_3x3, iterations=1)

        img_final_bin = ~grid_final_bin & bin_thresh

        kernel_2x2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_final_bin = cv2.dilate(img_final_bin, kernel_3x3, iterations=1)
        img_final_bin = cv2.erode(img_final_bin, kernel_3x3, iterations=1)

        img_final_bin = cv2.morphologyEx(img_final_bin, cv2.MORPH_OPEN, kernel_1x2, iterations=1)
        img_final_bin = cv2.morphologyEx(img_final_bin, cv2.MORPH_OPEN, kernel_2x1, iterations=1)

        kernel_2x2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_final_bin = cv2.erode(img_final_bin, kernel_2x2, iterations=1)

        img_final_bin = ~img_final_bin

        if self.is_debug:

            img_name = name if name is not None else 'img_final_bin'
            cv2.imwrite(os.path.join(self.debug_path, '{}.png'.format(img_name)), img_final_bin)

        return img_final_bin
        return img_final_bin
