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
        self.grab_and_cut_mask = self.grab_and_cut()

    def initialize_source_img(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        imgs_path = os.path.join(absolute_path, '*.jpg').strip()
        img_files = glob(imgs_path)

        return cv.imread(img_files[5])
    
    def initialize_target_img(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        imgs_path = os.path.join(absolute_path, '*.jpg').strip()
        img_files = glob(imgs_path)

        return cv.imread(img_files[8])
    
    def grab_and_cut(self):
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
    
    def get_grab_and_cut_mask(self):
        return self.grab_and_cut_mask
    
    def get_source_img(self):
        return self.source_img
    
    def get_target_img(self):
        return self.target_img

