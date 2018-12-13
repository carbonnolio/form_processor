import os
import pytesseract
import json
import cv2
import numpy as np

from difflib import SequenceMatcher
from PIL import Image
from pdf2image import convert_from_path

class ImagePreprocessor:

    def __init__(self, to_image_type='png', is_debug=False, form_dict_path=None, detect_threshold=0.6):

        self.to_image_type = to_image_type.lower()
        self.is_debug = is_debug
        self.detect_threshold = detect_threshold

        self.x_tol = 15
        self.y_tol = 15
        self.w_tol = 15
        self.h_tol = 5

        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

        if is_debug:
            self.debug_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Debug', 'ImagePreprocessor')
            
            if not os.path.exists(self.debug_path):
                os.makedirs(self.debug_path)

        if form_dict_path is not None:
            dict_path_file = open(form_dict_path)
            dict_path_str = dict_path_file.read()
            self.dict_path_json = json.loads(dict_path_str)
    
    def to_normal_type(self, file_path):
        
        _, file_extension = os.path.splitext(file_path)    
        file_type = file_extension[1:].lower()

        cv2img_list = []

        if file_type == 'pdf':
            images = convert_from_path(file_path)

            if self.is_debug:
                img_idx = 0

                for img in images:
                    img.save(os.path.join(self.debug_path, '{}.{}'.format(img_idx, self.to_image_type)), self.to_image_type)
                    img_idx += 1

            for img in images:
                cv2img = np.array(img)
                cv2img_list.append(cv2img)
            
        else:
            cv2img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            cv2img_list.append(cv2img)

        return cv2img_list

    def detect_form_type(self, img):

        img_w, img_h, _ = img.shape
        cropped_top = img[0:img_h // 6, 0:img_w]

        detected_str = pytesseract.image_to_string(cropped_top).lower()

        for form_type in self.dict_path_json:
            true_count = 0

            form_type_arr = form_type['value'].split(',')

            for word in form_type_arr:
                if word.strip().lower() in detected_str:
                    true_count += 1

            ratio = true_count / len(form_type_arr)

            if self.is_debug:
                print('Match ratio for the form {} is {}'.format(form_type['key'], ratio))
            
            if true_count/len(form_type_arr) >= self.detect_threshold:
                resized_img = cv2.resize(img, (form_type['w'], form_type['h']))
                
                if self.is_debug:
                    file_name = form_type['key'] + '.{}'.format(self.to_image_type)
                    save_path = os.path.join(self.debug_path, 'DetectedForms', file_name)

                    cv2.imwrite(save_path, resized_img)

                return form_type['key'], resized_img

        if self.is_debug:
            print('Failed to find form match in dict!')

        return None

    def get_grid_fields(self, img, form_type=None):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, img_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        kernel_6x10 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 10))

        im_floodfill = img_bin.copy()
        h, w = img_bin.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = img_bin | im_floodfill_inv

        im_out = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel_6x10, iterations=5)

        if self.is_debug:
            cv2.imwrite(os.path.join(self.debug_path, 'DetectedGrid', 'detected_grid.png'), im_out)

        _, contours, _ = cv2.findContours(im_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.is_debug:
            back_to_rgb = cv2.cvtColor(im_out, cv2.COLOR_GRAY2RGB)

            cv2.drawContours(back_to_rgb, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.debug_path, 'DetectedGrid', 'detected_grid_contoured.png'), back_to_rgb)

        features_imgs = []

        for c in contours:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if(len(approx) == 4):
                x, y, w, h = cv2.boundingRect(c)

                if form_type is not None:
                    for _, item in enumerate(self.dict_path_json):
                        if item['key'] == form_type:

                            for feature in item['features']:
                                if self.correct_item(feature, x, y, w, h):

                                    new_img = img[y: y+h, x: x+w]

                                    features_imgs.append((feature['type'], new_img))

                                    if self.is_debug:
                                        img_name = str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h) + '.png'
                                        cv2.imwrite(os.path.join(self.debug_path, 'DetectedFeatures', img_name), new_img)
                else:
                    new_img = img[y: y+h, x: x+w]
                    features_imgs.append(new_img)

        return features_imgs

    def correct_item(self, data, x, y, w, h):
        is_x = data['x']-self.x_tol <= x <= data['x']+self.x_tol
        is_y = data['y']-self.y_tol <= y <= data['y']+self.y_tol
        is_w = data['w']-self.w_tol <= w <= data['w']+self.w_tol
        is_h = data['h']-self.h_tol <= h <= data['h']+self.h_tol
        return is_x & is_y & is_w & is_h
