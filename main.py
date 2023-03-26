import image
import user
import numpy as np
import cv2 as cv
import utils


init = image.Image()
source_img = init.get_source_img()
target_img = init.get_target_img()
mask = init.get_grab_and_cut_mask()


cv.imshow('MASCARA GRAB AND CUT', mask)
cv.waitKey(0)

poisson_opencv_mask = init.poisson_editing(mask)
cv.imshow("RESULTADO POISSON MASCARA OPENCV", poisson_opencv_mask)
cv.waitKey(0)

cur_boundary = user.get_user_boundary()


weights = utils.energy_boundary_minimization(target_img, source_img, mask) 


cut_line_coordinates = utils.define_cut_line_coordinates(mask)
#print(cut_line_coordinates)


mascara_opt_contorno = utils.dijkstra_algorithm(cut_line_coordinates, source_img, target_img, mask, weights)
cv.imshow("MASCARA COM CONTORNO OTIMIZADO", mascara_opt_contorno)
cv.waitKey(0)

poisson_opt_mascara = init.poisson_editing(mascara_opt_contorno)
cv.imshow("RESULTADO POISSON MASCARA OTIMIZADA", poisson_opt_mascara)
cv.waitKey(0)
































'''
start_points = tuple(utils.define_cut_line_coordinates(mask, cur_boundary))
graph = utils.energy_boundary_minimization(target_img, source_img, mask)

print( "Tamanho da mascara:",  mask.shape[1], mask.shape[0])
print("Tamanho do grafo" , graph.shape[1], graph.shape[0])
print("Cur Boundary:" , cur_boundary[3] - cur_boundary[1], cur_boundary[2] - cur_boundary[0])


(x,y) = start_points[0]

for i in range(0, 100):
   graph[i][x] = 500000000000000
   graph[i][x+3] = 500000000000000
   graph[i][x+2] = 500000000000000
   graph[i][x+1] = 500000000000000
   graph[i][x+4] = 500000000000000
   graph[i][x+5] = 500000000000000
   graph[i][x-3] = 500000000000000
   graph[i][x-2] = 500000000000000
   graph[i][x-1] = 500000000000000
   graph[i][x-4] = 500000000000000
   graph[i][x-5] = 500000000000000



cost_list = []
  
for i in range(0, len(start_points)): 
   (x,y) = start_points[0]
   print(x,y)
   min, prev_point = utils.dijkstra_algorithm(graph, (x,y))   
   cost_list.append(min[128][255])   # 
  # cost_list.append(min[128][255])   # 
   print(min[128][255])
   path,show_path = utils.find_shortest_path(prev_point, (128, 255), mask)
   cv.imshow('sww' , show_path)
   cv.waitKey(0)
   break

cost_list.sort()
print(cost_list[0])

#path,show_path = utils.find_shortest_path(prev_point, (2,2), cur_boundary)
#print(path)
#cv.imshow('qee' , show_path)
#cv.waitKey(0)

# cv.imshow('Source image', )
# cv.imshow('Target image', )
# cv.waitKey(0)
# cv.destroyAllWindows()



  # shw2 = cv.rotate(show_path, cv.ROTATE_90_CLOCKWISE)
   #cv.imshow('sww' , show_path)
  # shw3 = cv.flip(shw2, 1)


'''
