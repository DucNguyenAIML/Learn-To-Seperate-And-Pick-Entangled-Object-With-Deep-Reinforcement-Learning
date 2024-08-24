import cv2
import numpy as np

img1 = cv2.imread("C:\\Users\\USER\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork - RebuildCam\\LearnToPickEntangledObjectWithDeepQNetwork - RebuildCam\\heightmap_diff\\000001.0.depth.png", 0)
img2 = cv2.imread("C:\\Users\\USER\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork - RebuildCam\\LearnToPickEntangledObjectWithDeepQNetwork - RebuildCam\\heightmap_diff\\000002.0.depth_aftergrasp.png", 0)

value_img1 = np.asmatrix(img1)
value_img2 = np.asmatrix(img2)

new_img1 = np.where(value_img1 == 4, 255, value_img1)
new_img2 = np.where(value_img2 == 4, 255, value_img2)

cv2.imshow("new_img1", new_img1)
cv2.imshow("new_img2", new_img2)

cv2.waitKey()
