import cv2 as cv
import numpy as np
import user
import math
import matplotlib.pyplot as plt
import image
import heapq
from glob import glob

TAM = 1000000000
class Coor:
    def __init__(self, x, y):
        self.x = x
        self.y = y  

def compute_k(target_img, source_img):
    cur_boundary = user.get_user_boundary()
    sum_p = 0
    
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
  
    cur_boundary = (267,78,793,738)
    k = compute_k(target_img, source_img)
    
    cost_matrix = np.full((cur_boundary[3], cur_boundary[2]), -1, np.int32)
    
    for x in range(0, cur_boundary[2]):
        for y in range(0, cur_boundary[3]):
            xx = cur_boundary[0] + x
            yy = cur_boundary[1] + y
            mask_pixel = mask[yy][xx]
            if mask_pixel != 255:
                cost_matrix[y][x] = pow(color_diff(target_img[yy][xx], source_img[yy][xx]) - k, 2)

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
           
    # return cut_line_fst_point[0]


def define_cut_line_coordinates(mask):

    cur_boundary = (267,78,793,738)

   # tendo achado o primeiro ponto de cut line, precisamos traçar a linha

    cut_line_fst_point = define_init_cut_line(mask, cur_boundary)
    print(cut_line_fst_point)
    
    cut_line_coordinates = []
    
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

    # mostra a linha

    
    cut_line_screen = np.full((mask.shape), 0, np.uint8)
    for coordinate in cut_line_coordinates:
        x , y  = coordinate
        cut_line_screen[y][x] = 255

    for y in range(mask.shape[1]):
        for x in range(mask.shape[0]):
         if mask[x][y] == 255:
            cut_line_screen[x][y] = 255
    
    
    # imprime a cut line, representando as coordenadas de inicio dos caminhos no grafo

    return cut_line_coordinates


def dijkstra_algorithm(cut_line_coordinates, source_img, target_img, mask, weights):
    
    rect = (267,78,793,738)


    paths_list = np.full((len(cut_line_coordinates), rect[3], rect[2]), TAM, np.int64)
    parents_list = np.full((len(cut_line_coordinates), rect[3], rect[2]), Coor(0, 0), Coor)

    for i in range(0, len(cut_line_coordinates)):
        init = cut_line_coordinates[i]
      #  min_path(weights, paths_list[i], parents_list[i], cut_line_coordinates, init[0], init[1])
        visited = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
        calculated = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
        paths_list[i][init[1], init[0]] = 0
        visited[init[1], init[0]] = True
        calculated[init[1], init[0]] = True
        queue = []
        append_neighborhood(init[0], init[1], queue, weights, visited, cut_line_coordinates)
        while len(queue) > 0:
          [x, y] = queue.pop(0)
          node_weight = weights[y][x]
          neighborhood = get_neighborhood(x, y, weights, visited, cut_line_coordinates, True, calculated)
          paths_list[i][y][x] = node_weight + calc_min_weight(neighborhood, paths_list[i], parents_list[i], x, y)
          calculated[y][x] = True
          append_neighborhood(x, y, queue, weights, visited, cut_line_coordinates)


    # Encontra o menor caminho entre um pixel de destino e um pixel inicial nos resultados calculados
    min_weight = [0, 0, 0, 500000000000]
    for bi, cl in enumerate(cut_line_coordinates):
        for ed in range(-1, 2):
            [x, y] = cl
            end_x = x + 1
            end_y = y + ed
            if 0 <= end_y < len(paths_list[bi]):
                weight = paths_list[bi][end_y][end_x]
                if weight < min_weight[3]:
                    min_weight = [end_x, end_y, bi, weight]

    path = paths_list[min_weight[2]]
    parent_list = parents_list[min_weight[2]]
    selected_boundary = np.full((rect[3], rect[2]), 0, np.uint8)
    current_x = min_weight[0]
    current_y = min_weight[1]
    while path[current_y][current_x] != 0:
        selected_boundary[current_y][current_x] = 255
        next_node = parent_list[current_y][current_x]
        current_y = next_node.y
        current_x = next_node.x

    selected_boundary[current_y][current_x] = 255
    initial_inside_pixel = [0, 0]
    found = False

    #cv.imwrite("results/" + source_name + "/optimized_boundary.jpg", selected_boundary)
    #cv.imshow("Optimized Boundary", selected_boundary)
    #cv.waitKey(0)

    for y, row in enumerate(selected_boundary):
        for x, pixel in enumerate(row):
            if pixel == 0 and x - 1 > 0 and y > 0:
                last_pixel = selected_boundary[y][x - 1]
                up_pixel = selected_boundary[y - 1][x]
                if last_pixel == 255:
                    if up_pixel == 255:
                        initial_inside_pixel = [x, y]
                        found = True
                        break
        if found:
            break
       
    fill(selected_boundary, initial_inside_pixel)
    print(selected_boundary, initial_inside_pixel)

    optimized_boundary_mask = np.full((mask.shape[0], mask.shape[1]), 0, np.uint8)

    for y, row in enumerate(selected_boundary):
        for x, pixel in enumerate(row):
            original_y = y + rect[1]
            original_x = x + rect[0]
            optimized_boundary_mask[original_y][original_x] = pixel

    #cv.imwrite("results/" + source_name + "/optimized_boundary_mask.jpg", optimized_boundary_mask)
   # cv.imshow("Optimized Boundary Mask", optimized_boundary_mask)
   # cv.waitKey(0)

    return optimized_boundary_mask


def fill(selected_boundary, initial_inside_pixel):
    queue = [initial_inside_pixel]
    height = len(selected_boundary)
    width = len(selected_boundary[0])

    while len(queue) > 0:
        [x, y] = queue.pop(0)

        if width > x > 0 and height > y > 0 and selected_boundary[y][x] == 0:
            selected_boundary[y][x] = 255
            queue.append([x - 1, y])
            queue.append([x + 1, y])
            queue.append([x, y - 1])
            queue.append([x, y + 1])

'''
def min_path(weights, paths, parent_list, beginnings, init_x, init_y):
    visited = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
    calculated = np.full((weights.shape[0], weights.shape[1]), False, dtype=bool)
    paths[init_y, init_x] = 0
    visited[init_y, init_x] = True
    calculated[init_y, init_x] = True
    queue = []
    append_neighborhood(init_x, init_y, queue, weights, visited, beginnings)
    while len(queue) > 0:
        [x, y] = queue.pop(0)
        node_weight = weights[y][x]
        neighborhood = get_neighborhood(x, y, weights, visited, beginnings, True, calculated)
        paths[y][x] = node_weight + calc_min_weight(neighborhood, paths, parent_list, x, y)
        calculated[y][x] = True
        append_neighborhood(x, y, queue, weights, visited, beginnings)
'''

def append_neighborhood(x, y, queue, weights, visited, beginnings):
    queue.extend(get_neighborhood(x, y, weights, visited, beginnings))


def get_neighborhood(x, y, weights, visited, beginnings, calculated_neighborhood=False, calculated=None):
    if calculated is None:
        calculated = []
    neighborhood = []
    its_initial_coordinate = coordinate_belongs_to_list(x, y, beginnings)
    its_destiny_coordinate = coordinate_belongs_to_list(x - 1, y, beginnings)

    def validate_direction(target_x, target_y):
        column_size = len(weights)
        row_size = len(weights[0])
        if not is_safe(target_x, target_y, column_size, row_size) or weights[target_y][target_x] == -1:
            return [False, 0, 0]

        if its_initial_coordinate and coordinate_belongs_to_list(target_x - 1, target_y, beginnings):
            return [False, 0, 0]

        if its_destiny_coordinate and coordinate_belongs_to_list(target_x, target_y, beginnings):
            return [False, 0, 0]

        if calculated_neighborhood:
            valid_direction = calculated[target_y][target_x]
        else:
            valid_direction = not visited[target_y][target_x]

        return [valid_direction, target_x, target_y]

    top_y = y - 1
    bottom_y = y + 1
    left_x = x - 1
    right_x = x + 1

    directions = [
        validate_direction(left_x, y),
        validate_direction(right_x, y),
        validate_direction(x, top_y),
        validate_direction(x, bottom_y),
        validate_direction(left_x, top_y),
        validate_direction(left_x, bottom_y),
        validate_direction(right_x, top_y),
        validate_direction(right_x, bottom_y)
    ]

    for direction in directions:
        is_valid = direction[0]
        if is_valid:
            x = direction[1]
            y = direction[2]
            visited[y][x] = True
            neighborhood.append([x, y])

    return neighborhood


def coordinate_belongs_to_list(target_x, target_y, coordinates_list):
    for i in coordinates_list:
        [x, y] = i
        if target_x == x and target_y == y:
            return True
    return False


def is_safe(x, y, column_size, row_size):
    return row_size > x >= 0 and column_size > y >= 0


def calc_min_weight(paths_coordinates, paths, parent_list, x, y):
    min_weight = 50000000000
    valid_coordinates = []
    for path_coordinates in paths_coordinates:
        if len(paths) > path_coordinates[1] >= 0 and len(paths[0]) > path_coordinates[0] >= 0:
            valid_coordinates.append(path_coordinates)
    for coordinate in valid_coordinates:
        [dest_x, dest_y] = coordinate
        weight = paths[dest_y][dest_x]
        if weight < min_weight:
            min_weight = weight
            parent_list[y][x] = Coor(dest_x, dest_y)
    return min_weight


    



























    '''
    cv.imshow("Cut line" , cut_line_screen)
    cv.waitKey(0)
    
    
    limit_line = np.full((cur_boundary[3], cur_boundary[2]), 0, np.uint8)
    for y in range(cur_boundary[1], cur_boundary[1] + cur_boundary[3]):
        for x in range(cur_boundary[0], cur_boundary[0] + cur_boundary[2]):
            limit_line[y - cur_boundary[1]][x - cur_boundary[0]] = mask[y][x]

    for beginning in cut_line_coordinates:
        (bx, by) = beginning
        limit_line[by][bx] = 255
    '''









































'''                   # weights
def dijkstra_algorithm(graph, start_point):
    # initialize graphs to track if a point is visited,
    # current calculated distance from start to point,
    # and previous point taken to get to current point
    visited = [[False for col in row] for row in graph]
    distance = [[float('inf') for col in row] for row in graph]
    distance[start_point[0]][start_point[1]] = 0
    prev_point = [[None for col in row] for row in graph]
    n, m = len(graph), len(graph[0])
    number_of_points, visited_count = n * m, 0
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    min_heap = []

    # min_heap item format:
    # (pt's dist from start on this path, pt's row, pt's col)
    heapq.heappush(min_heap, (distance[start_point[0]][start_point[1]], start_point[0], start_point[1]))

    while visited_count < number_of_points:
 
        current_point = heapq.heappop(min_heap)
        distance_from_start, row, col = current_point
                          
        for direction in directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if -1 < new_row < n and -1 < new_col < m and not visited[new_row][new_col]:
                dist_to_new_point = distance_from_start + graph[new_row][new_col]
                if dist_to_new_point < distance[new_row][new_col]:
                    distance[new_row][new_col] = dist_to_new_point
                    prev_point[new_row][new_col] = (row, col)
                    heapq.heappush(min_heap, (dist_to_new_point, new_row, new_col))
        visited[row][col] = True
        visited_count += 1

    return distance, prev_point
'''





















































'''
def find_shortest_path(prev_point_graph, end_point, mask):
#def find_shortest_path(prev_point_graph, end_point, cur_boundary):

   # show_path = np.full((cur_boundary[3], cur_boundary[2]), 0, np.uint8)
   # show_path = np.full((mask.shape[1], mask.shape[0]), 0, np.uint8)
    show_path = np.full(mask.shape, 0, np.uint8)

    # preenche com a parte branca da máscara

    
    for y in range(mask.shape[1]):      
      for x in range(mask.shape[0]):
        if mask[x][y] == 255: 
            show_path[x][y] = 255
    

    shortest_path = []
    
    current_point = end_point
    while current_point is not None:
        shortest_path.append(current_point)
        current_point = prev_point_graph[current_point[0]][current_point[1]]
        if current_point is not None:
        # print(current_point)
        # if current_point[1] < 844:           
           show_path[current_point[0]][current_point[1]] = 255
    shortest_path.reverse()

    return shortest_path, show_path
'''


'''
def find_shortest_path(prev_point_graph, end_point, mask):
#def find_shortest_path(prev_point_graph, end_point, cur_boundary):

   # show_path = np.full((cur_boundary[3], cur_boundary[2]), 0, np.uint8)
    show_path = np.full((mask.shape[1], mask.shape[0]), 0, np.uint8)

    # preenche com a parte branca da máscara

    
    for x in range(mask.shape[0]):
      for y in range(mask.shape[1]):
        if mask[x][y] == 255: 
            show_path[y][x] = 255


    shortest_path = []
    
    current_point = end_point
    while current_point is not None:
        shortest_path.append(current_point)
        current_point = prev_point_graph[current_point[0]][current_point[1]]
        if current_point is not None:
         show_path[current_point[1]][current_point[0]] = 255
    shortest_path.reverse()

    return shortest_path, show_path
'''
