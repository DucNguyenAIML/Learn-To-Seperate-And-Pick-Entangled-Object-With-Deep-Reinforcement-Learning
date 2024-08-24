import cv2

img = cv2.imread("C:\\Users\\USER\\Desktop\\LearnToPickEntangledObjectWithDeepQNetwork - RebuildCam\\LearnToPickEntangledObjectWithDeepQNetwork - RebuildCam\\aftergrasp_log\\color_aftergrasp.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Window", img)

cv2.waitKey()