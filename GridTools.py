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


    def find_crossings(self, image_path):

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bin_img = cv2.bitwise_not(gray)

        kernel_1x2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        kernel_2x1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 55))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))

        img_temp1 = cv2.erode(bin_img, ver_kernel, iterations=3)
        ver_lines_img = cv2.dilate(img_temp1, ver_kernel, iterations=3)
        # cv2.imwrite(os.path.join(self.debug_path, 'ver_lines_img.png'), ver_lines_img)

        img_temp2 = cv2.erode(bin_img, hor_kernel, iterations=3)
        hor_lines_img = cv2.dilate(img_temp2, hor_kernel, iterations=3)
        # cv2.imwrite(os.path.join(self.debug_path, 'hor_lines_img.png'), hor_lines_img)

        # if self.is_debug:
            # cv2.imwrite(os.path.join(self.debug_path, 'original_gray.png'), gray)

        merged_lines = cv2.addWeighted(ver_lines_img, 0.5, hor_lines_img, 0.5, 0.0)

        _, grid_final_bin = cv2.threshold(merged_lines, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        grid_final_bin = cv2.dilate(grid_final_bin, kernel_3x3, iterations=1)

        # cv2.imwrite(os.path.join(self.debug_path, 'grid_final_bin.png'), grid_final_bin)

        img_final_bin = ~grid_final_bin & bin_img

        img_final_bin = cv2.dilate(img_final_bin, kernel_3x3, iterations=1)
        img_final_bin = cv2.erode(img_final_bin, kernel_3x3, iterations=1)

        img_final_bin = cv2.morphologyEx(img_final_bin, cv2.MORPH_OPEN, kernel_1x2, iterations=1)
        img_final_bin = cv2.morphologyEx(img_final_bin, cv2.MORPH_OPEN, kernel_2x1, iterations=1)

        cv2.imwrite(os.path.join(self.debug_path, 'img_final_bin.png'), ~img_final_bin)

        # edges = cv2.Canny(bin_img, 75, 150)

        # cv2.imwrite(os.path.join(self.debug_path, 'edges.png'), edges)

        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=30)

        # back_to_rgb = cv2.cvtColor(~img_final_bin, cv2.COLOR_GRAY2RGB)

        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(back_to_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # cv2.imwrite(os.path.join(self.debug_path, 'back_to_rgb.png'), back_to_rgb)

        # kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # gray = cv2.erode(gray, kernel3x3, iterations=1)

        # blur = cv2.medianBlur(img_final_bin, 5)
        # bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 8)

        # if self.is_debug:
            # cv2.imwrite(os.path.join(self.debug_path, 'gaussian_bin.png'), bin_img)

        

        # rho, theta, thresh = 2, np.pi/180, 100
        # lines = cv2.HoughLines(bin_img, rho, theta, thresh)
        
        if self.is_debug:
            back_to_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)

            # for i in range(0, len(lines)):
            #     for x1, y1, x2, y2 in lines[i]:
            #         cv2.line(back_to_rgb, (x1, y1), (x2, y2), (0,255, 0), 2)
            
            # cv2.imwrite(os.path.join(self.debug_path, 'with_hough_lines.png'), back_to_rgb)

