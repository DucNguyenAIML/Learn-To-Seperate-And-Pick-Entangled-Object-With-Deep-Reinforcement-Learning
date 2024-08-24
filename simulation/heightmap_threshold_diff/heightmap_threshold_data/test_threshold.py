import cv2
import numpy as np
import os
Height_map = cv2.imread("C:\\Users\\gplmi\\OneDrive\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork\\simulation\\heightmap_threshold_diff\\heightmap_threshold_data\\000001.0.depth.png", 0)

print(Height_map)
matrix_value = np.asmatrix(Height_map)
sum = np.matrix.sum(matrix_value)
print(sum)
print(matrix_value.shape)

with open('C:\\Users\\gplmi\\OneDrive\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork\\simulation\\heightmap_threshold_diff\\heightmap_threshold_data\\Matrix_Value.txt', 'w') as f:
    for row in matrix_value:
        f.write(' '.join([str(a) for a in row]) + '\n')

new_matrix_value = np.where(matrix_value == 14, 255, matrix_value)
IMG = np.array(new_matrix_value ,dtype=np.uint8)
# IMG = cv2.imread(new_matrix_value)
cv2.imshow("IMG", IMG)
cv2.waitKey(0)
# print(os.listdir("C:\\Users\\gplmi\\OneDrive\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork\\logs\\2024-06-22.20-52-43\\data\\color-heightmaps"))
# print(type(os.listdir("C:\\Users\\gplmi\\OneDrive\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork\\logs\\2024-06-22.20-52-43\\data\\color-heightmaps")))