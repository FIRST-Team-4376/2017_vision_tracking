import cv2
import numpy as np
#from matplotlib import pyplot as plt


rgb_min_blue = np.uint8([[[50, 50, 200]]])
rgb_max_blue = np.uint8([[[100,0,255 ]]])

# hsv_min = np.array([110,50,50])
# hsv_max = np.array([130,255,255])
hsv_max = cv2.cvtColor(rgb_max_blue,cv2.COLOR_BGR2HSV)
hsv_min = cv2.cvtColor(rgb_min_blue,cv2.COLOR_BGR2HSV)


# hsv_max = np.array([130,30,30])
# hsv_min = np.array([130,255,255])

cap = cv2.VideoCapture(1)


ret, img = cap.read()
# img = cv2.imread('122_H_0deg.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img, hsv_min, hsv_max)

res = cv2.bitwise_and(img,img, mask= mask)

# plt.imshow(mask)
# plt.imshow(res)
plt.imshow(res)
plt.show()



# TODO: I think we might have to upgrade opencv for this to work
# cv2.imshow('mask', mask)
# cv2.imshow('res', res)

# k = cv2.waitKey(5) & 0xFF
# if k == 27:
#   break



# plt.imshow(img)
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
# plt.show()
