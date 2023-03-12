import image
import user
import numpy as np
import cv2 as cv
import utils

init = image.Image()
source_img = init.get_source_img()
target_img = init.get_target_img()
mask = init.get_grab_and_cut_mask()


cur_boundary = user.get_user_boundary()
cut_line_coordinates = utils.define_cut_line_coordinates(mask, cur_boundary)
print(cut_line_coordinates)

#utils.energy_boundary_minimization(init.get_target_img(), init.get_source_img(), init.get_grab_and_cut_mask())


# cv.imshow('Source image', )
# cv.imshow('Target image', )
# cv.waitKey(0)
# cv.destroyAllWindows()
