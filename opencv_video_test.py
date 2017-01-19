import numpy as np
import cv2
# from matplotlib import pyplot as plt

# rgb_min_blue = np.uint8([[[27, 60, 40]]])
# rgb_max_blue = np.uint8([[[81,255,120 ]]])
# hsv_max = cv2.cvtColor(rgb_max_blue,cv2.COLOR_BGR2HSV)
# hsv_min = cv2.cvtColor(rgb_min_blue,cv2.COLOR_BGR2HSV)

# green tape

def nothing(x):
    pass
hmin = 40
hmax = 150

smin = 40
smax = 255

vmin = 150
vmax = 255

#blue
# hsv_min = np.array([110,50,50])
# hsv_max = np.array([130,255,255])


contours_arg_1 = 0
contours_arg_3 = 3

cv2.namedWindow("controls", cv2.WINDOW_NORMAL | cv2.WINDOW_OPENGL)
cv2.createTrackbar('Hmin','controls', hmin, 255, nothing)
cv2.createTrackbar('Hmax','controls', hmax, 255, nothing)
cv2.createTrackbar('Smin','controls', smin, 255, nothing)
cv2.createTrackbar('Smax','controls', smax, 255, nothing)
cv2.createTrackbar('Vmin','controls', vmin, 255, nothing)
cv2.createTrackbar('Vmax','controls', vmax, 255, nothing)
cv2.createTrackbar('CountoursArg1','controls', contours_arg_1, 25, nothing)
cv2.createTrackbar('CountoursArg3','controls', contours_arg_3, 25, nothing)


# !!! USEFUL !!!
# http://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
# cv2.imshow('controls',small)

while(True):

    hsv_min = np.array([hmin,smin,vmin])
    hsv_max = np.array([hmax,smax,vmax])
    ret, img = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
    # greyscale_image = mask

    masked_image = cv2.bitwise_and(img,img, mask= mask)

    # gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    # corners = cv2.goodFeaturesToTrack(gray, 100, .35, 15)
    # if corners is None:
    #     print("no corners!")
    # else:
    #     corners = np.int0(corners)
    #     for corner in corners:
    #         x,y = corner.ravel()
    #         cv2.circle(masked_image,(x,y),5,255,-1)

    ret2, thresh = cv2.threshold(mask, 40,40,150)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, contours_arg_1 - 1, (0,255,0), contours_arg_3)


    # imshow doesnt work on mac for some reason

    # cv2.rectangle(masked_image,(15,20),(70, 50), ( 0, 55, 255), 2)

    small = cv2.resize(masked_image, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('frame',small)
    # plt.imshow(gray)
    # plt.show()


    hmin = cv2.getTrackbarPos ('Hmin', 'controls')
    # print(hmin)
    hmax = cv2.getTrackbarPos ('Hmax', 'controls')
    # print(hmax)
    smin = cv2.getTrackbarPos ('Smin', 'controls')
    smax = cv2.getTrackbarPos ('Smax', 'controls')
    vmin = cv2.getTrackbarPos ('Vmin', 'controls')
    vmax = cv2.getTrackbarPos ('Vmax', 'controls')
    contours_arg_1 = cv2.getTrackbarPos ('CountoursArg1', 'controls')
    contours_arg_3 = cv2.getTrackbarPos ('CountoursArg3', 'controls')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
