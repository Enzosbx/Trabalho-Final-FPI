import os
import cv2 as cv
from glob import glob
import numpy as np
import user
import math

class Image:

    def __init__(self):
        self.source_img = self.initialize_source_img()
        self.target_img = self.initialize_target_img()

    def initialize_source_img(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        imgs_path = os.path.join(absolute_path, '*.jpg').strip()
        img_files = glob(imgs_path)

        return cv.imread(img_files[1])
    
    def initialize_target_img(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        imgs_path = os.path.join(absolute_path, '*.jpg').strip()
        img_files = glob(imgs_path)

        return cv.imread(img_files[7])
    
    def grab_and_Cut(self):
        mask = np.zeros(self.source_img.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        rect = user.user_drawn_boundary(self.source_img)
        
        mask, bgdModel, fgdModel = cv.grabCut(self.source_img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        
        outputMask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),0, 1)

        outputMask = (outputMask * 255).astype("uint8")

        return outputMask

    def poisson_editing(self, mask):
        width, height, channels = self.target_img.shape

        center = (round(height / 2), round(width / 2)) # center of source image at center of target image

        mixed_clone = cv.seamlessClone(self.source_img, self.target_img, mask, center, cv.MIXED_CLONE)

        return mixed_clone
    
    def compute_k(self):
        cur_boundary = user.get_user_boundary()
        sum_p = 0

        #print(f'left: {cur_boundary[0]}, top: {cur_boundary[1]}, right: {cur_boundary[2]}, bottom: {cur_boundary[3]}')
        for x in range(cur_boundary[0], cur_boundary[2]):
            sum_p += self.color_diff(self.target_img[cur_boundary[3]][x], self.source_img[cur_boundary[3]][x])
            sum_p += self.color_diff(self.target_img[cur_boundary[3]][x], self.source_img[cur_boundary[3]][x])

        for y in range(cur_boundary[1], cur_boundary[3]):
            sum_p += self.color_diff(self.target_img[y][cur_boundary[2]], self.source_img[y][cur_boundary[2]])
            sum_p += self.color_diff(self.target_img[y][cur_boundary[0]], self.source_img[y][cur_boundary[0]])
        
        perimeter = ((2*cur_boundary[3]-cur_boundary[0])+(2*cur_boundary[2]-cur_boundary[1]))

        return (1/perimeter) * sum_p
        
    
    def color_diff(self, target, source):
        return math.sqrt(pow(int(target[0]) - int(source[0]), 2) +
                        pow(int(target[1]) - int(source[1]), 2) + 
                        pow(int(target[2]) - int(source[2]), 2))


