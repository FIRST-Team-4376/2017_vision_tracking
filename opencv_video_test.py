import numpy as np
import cv2
from matplotlib import pyplot as plt

# rgb_min_blue = np.uint8([[[27, 60, 40]]])
# rgb_max_blue = np.uint8([[[81,255,120 ]]])
# hsv_max = cv2.cvtColor(rgb_max_blue,cv2.COLOR_BGR2HSV)
# hsv_min = cv2.cvtColor(rgb_min_blue,cv2.COLOR_BGR2HSV)

# green tape
hsv_min = np.array([70,50,40])
hsv_max = np.array([100,255,255])

#blue
# hsv_min = np.array([110,50,50])
# hsv_max = np.array([130,255,255])


cap = cv2.VideoCapture(1)
while(True):
    ret, img = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, hsv_min, hsv_max)

    res = cv2.bitwise_and(img,img, mask= mask)
 
    # imshow doesnt work on mac for some reason
    cv2.imshow('frame',res)
    # plt.imshow(gray)
    # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
