import image
import user
import numpy as np
import cv2 as cv
import utils

init = image.Image()
source_img = init.get_source_img()
target_img = init.get_target_img()
utils.energy_boundary_minimization(init.get_target_img(), init.get_source_img(), init.get_grab_and_cut_mask())
# cv.imshow('Source image', )
# cv.imshow('Target image', )
# cv.waitKey(0)
# cv.destroyAllWindows()

