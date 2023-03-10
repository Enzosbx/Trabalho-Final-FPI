import cv2 as cv
import numpy as np
import user
import math
import matplotlib.pyplot as plt

def compute_k(target_img, source_img):
    cur_boundary = user.get_user_boundary()
    sum_p = 0
    
    #print(f'left: {cur_boundary[0]}, top: {cur_boundary[1]}, right: {cur_boundary[2]}, bottom: {cur_boundary[3]}')
    for x in range(cur_boundary[0], cur_boundary[2]):
        sum_p += color_diff(target_img[cur_boundary[3]][x], source_img[cur_boundary[3]][x])
        sum_p += color_diff(target_img[cur_boundary[3]][x], source_img[cur_boundary[3]][x])

    for y in range(cur_boundary[1], cur_boundary[3]):
        sum_p += color_diff(target_img[y][cur_boundary[2]], source_img[y][cur_boundary[2]])
        sum_p += color_diff(target_img[y][cur_boundary[0]], source_img[y][cur_boundary[0]])
    
    perimeter = ((2*cur_boundary[3]-cur_boundary[0])+(2*cur_boundary[2]-cur_boundary[1]))

    return (1/perimeter) * sum_p

def color_diff(target, source):
    return math.sqrt(pow((int(target[0]) - int(source[0])), 2) +
                    pow((int(target[1]) - int(source[1])), 2) + 
                    pow((int(target[2]) - int(source[2])), 2))

def energy_boundary_minimization(target_img, source_img, mask):
    cur_boundary = user.get_user_boundary()
    k = compute_k(target_img, source_img)
    cost_matrix = np.zeros((mask.shape), dtype='uint8')
    cv.imshow('mask', mask)

    for y in range(mask.shape[1]):
        for x in range(mask.shape[0]):
            if mask[x][y] != 255: 
                cost_matrix[x][y] = pow((color_diff(target_img[x][y], source_img[x][y]) - k),2)

    cv.imshow('cost_matrix', cost_matrix)
    cv.waitKey(0)
    cv.destroyAllWindows()