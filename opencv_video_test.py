import numpy as np
import cv2
from matplotlib import pyplot as plt

# rgb_min_blue = np.uint8([[[27, 60, 40]]])
# rgb_max_blue = np.uint8([[[81,255,120 ]]])
# hsv_max = cv2.cvtColor(rgb_max_blue,cv2.COLOR_BGR2HSV)
# hsv_min = cv2.cvtColor(rgb_min_blue,cv2.COLOR_BGR2HSV)

# green tape
hsv_min = np.array([40,40,10])
hsv_max = np.array([120,255,255])

#blue
# hsv_min = np.array([110,50,50])
# hsv_max = np.array([130,255,255])


cap = cv2.VideoCapture(1)
while(True):
    ret, img = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, hsv_min, hsv_max)

    masked_image = cv2.bitwise_and(img,img, mask= mask)

    gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.10, 10)
    corners = np.int0(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(masked_image,(x,y),5,255,-1)
 
    # imshow doesnt work on mac for some reason
    cv2.imshow('frame',masked_image)
    # plt.imshow(gray)
    # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
