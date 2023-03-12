import cv2 as cv
import numpy as np
import user
import math
import matplotlib.pyplot as plt
import image

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
    return cost_matrix

# encontra o primeiro ponto válido da máscara
def define_init_cut_line(mask, cur_boundary):

    cut_line_fst_point = [[-1,-1], [-1,-1], [-1,-1], [-1,-1]]

    found_first_p = False

    for x in range(cur_boundary[0], cur_boundary[2]):
        for y in range(cur_boundary[1], cur_boundary[3]):
            pixel = mask[y][x]
            if pixel == 255:
                cut_line_fst_point[0] = [x, y]
                found_first_p = True
                break
        if found_first_p:
            break

    found_first_p = False

    for y in range(cur_boundary[1], cur_boundary[3]):
        for x in range(cur_boundary[0], cur_boundary[2]):
            pixel = mask[y][x]
            if pixel == 255:
                cut_line_fst_point[1] = [x, y]
                found_first_p = True
                break
        if found_first_p:
            break

    found_first_p = False

    x_list = []
    for i in cut_line_fst_point:
        x_list.append(i[0])

    y_list = []
    for i in cut_line_fst_point:
        y_list.append (i[1])

    x_list.sort()
    closer_x = 0
    if (x_list[0] - cur_boundary[0] >= cur_boundary[2] - x_list[-1]):
        closer_x = x_list[0]
        x = x_list[0]
    else:
        closer_x = x_list[2]
        x = x_list[2]

    y_list.sort()

    closer_y = 0
    if (y_list[0] - cur_boundary[1] >= cur_boundary[3] - y_list[-1]):
        closer_y = x_list[0]
        y = y_list[0]
    else:
        closer_y = x_list[2]
        y = y_list[2]
    
    if closer_x < closer_y:
        for par in cut_line_fst_point:
            if par[0] == x:
                return par  # par = fst_point
    else:
        for par in cut_line_fst_point:
           if par[1] == y:
                return par  # par = fst_point

    return cut_line_fst_point[0]


def define_cut_line_coordinates(mask, cur_boundary):

   # tendo achado o primeiro ponto de cut line, precisamos traçar a linha

    cut_line_fst_point = define_init_cut_line(mask, cur_boundary)
    print(cut_line_fst_point)
    
    cut_line_coordinates = []

    print(cut_line_fst_point[0])
    print(cur_boundary[0])
    print(cut_line_fst_point[1])
    print(cur_boundary[1])
    
    if  (cut_line_fst_point[0] - cur_boundary[0]) >= (cut_line_fst_point[1] - cur_boundary[1]):
        for point_y in range(cur_boundary[1], cut_line_fst_point[1]):  # mais próximo eixo y
            x = cut_line_fst_point[0] - cur_boundary[0]
            y = point_y - cur_boundary[1]
            cut_line_coordinates.append([x,y])
    else:
        for point_x in range(cur_boundary[0], cut_line_fst_point[0]):  # mais próximo eixo x
            x = point_x - cur_boundary[0]
            y = cut_line_fst_point[1] - cur_boundary[1]
            cut_line_coordinates.append([x,y])     

    return cut_line_coordinates
    
    
