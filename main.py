import image
import user
import numpy as np
import cv2 as cv

source_img = image.Image().source_img
target_img = image.Image().target_img

# cv.imshow('Source image', source_img)
# cv.imshow('Target image', target_img)

mask = image.Image().grab_and_Cut()

cv.imshow('a', mask)
cv.waitKey(0)


poisson_result = image.Image().poisson_editing(mask)
cv.imshow('r', user.user_drawn_boundary(poisson_result))
cv.waitKey(0)

