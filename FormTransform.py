import cv2
import os
import numpy as np

class FormTransform:

    def __init__(self, max_features, good_match, is_debug=False):

        self.max_features = max_features
        self.good_match = good_match
        self.is_debug = is_debug

        if is_debug == True:
            self.debug_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Debug', 'FormTransform')
            
            if not os.path.exists(self.debug_path):
                os.makedirs(self.debug_path)

    def align_image(self, im1, im2):

        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(self.max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        matches.sort(key=lambda x: x.distance, reverse=False)

        numGoodMatches = int(len(matches) * self.good_match)
        matches = matches[:numGoodMatches]

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        height, width, _ = im2.shape
        corrected = cv2.warpPerspective(im1, matrix, (width, height), cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        if self.is_debug:
            cv2.imwrite(os.path.join(self.debug_path, 'fixed.png'), corrected)

        return corrected