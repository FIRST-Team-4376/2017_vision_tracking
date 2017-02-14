import numpy as np
import cv2

# rgb_min_blue = np.uint8([[[27, 60, 40]]])
# rgb_max_blue = np.uint8([[[81,255,120 ]]])
# hsv_max = cv2.cvtColor(rgb_max_blue,cv2.COLOR_BGR2HSV)
# hsv_min = cv2.cvtColor(rgb_min_blue,cv2.COLOR_BGR2HSV)

# green tape

def nothing(x):
    pass


def width_and_height_from_contour(contour, image_to_draw_on, approximation_value):
    epsilon = approximation_value * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    x,y,w,h = cv2.boundingRect(approx)
    # cv2.rectangle(image_to_draw_on,(x,y),(x+w, y+h), (0,0,255), 4)
    return [w, h]





def draw_bounding_rectangle(image_to_draw_on, contours, approximation_value):

    the_thing_we_want = contours[0]
    lowest_difference = 99999999999999999999999

    for found_contour in contours:
        width_and_height = width_and_height_from_contour(found_contour, image_to_draw_on, approximation_value)

        difference = abs((width_and_height[0] / width_and_height[1]) - (2 / 5))
        if difference < lowest_difference:
            lowest_difference = difference
            the_thing_we_want = found_contour



    epsilon = approximation_value * cv2.arcLength(the_thing_we_want, True)
    approx = cv2.approxPolyDP(the_thing_we_want, epsilon, True)
    x,y,w,h = cv2.boundingRect(approx)
    cv2.rectangle(image_to_draw_on,(x,y),(x+w, y+h), (0,0,255), 4)

    print "width: "
    print width_and_height_from_contour(the_thing_we_want, image_to_draw_on, approximation_value)[0]
    print "height: "
    print width_and_height_from_contour(the_thing_we_want, image_to_draw_on, approximation_value)[1]

    # largest_contour = contours[0]
    # for found_contour in contours:

    #     if cv2.contourArea(found_contour) < cv2.contourArea(largest_contour):
    #         largest_contour = found_contour



    # print cv2.contourArea(largest_contour)
    # epsilon = approximation_value * cv2.arcLength(largest_contour, True)
    # approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    # # cv2.drawContours(image_to_draw_on, [approx], -1, (255,0,0), 4)
    # if len(approx) == 4:
    #     x,y,w,h = cv2.boundingRect(approx)
    #     cv2.rectangle(image_to_draw_on,(x,y),(x+w, y+h), (0,0,255), 4)



    # cv2.drawContours(image_to_draw_on, [found_contour], -1, (0,255,0), 4) 

hmin = 40
hmax = 150
#blue
# hsv_min = np.array([110,50,50])
# hsv_max = np.array([130,255,255])


hmin = 40
hmax = 150

smin = 40
smax = 255

vmin = 150
vmax = 255

blur_factor = 25

cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar('blur_factor','controls', blur_factor, 50, nothing)
cv2.createTrackbar('Hmin','controls', hmin, 255, nothing)
cv2.createTrackbar('Hmax','controls', hmax, 255, nothing)
cv2.createTrackbar('Smin','controls', smin, 255, nothing)
cv2.createTrackbar('Smax','controls', smax, 255, nothing)
cv2.createTrackbar('Vmin','controls', vmin, 255, nothing)
cv2.createTrackbar('Vmax','controls', vmax, 255, nothing)


cap = cv2.VideoCapture(1)
ret, img = cap.read()
small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.imshow('controls',small)

while(True):

    hsv_min = np.array([hmin,40,150])
    hsv_max = np.array([hmax,255,260])
    ret, img = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
    while (blur_factor % 2 != 1):
        blur_factor += 1
    mask = cv2.medianBlur(mask,blur_factor)
    greyscale_image = mask

    masked_image = cv2.bitwise_and(img,img, mask= mask)

    ret2, thresh = cv2.threshold(greyscale_image, 40,40,150)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(masked_image, contours, -1, (0,255,0), 3)
    draw_bounding_rectangle(masked_image, contours, .010)


    # imshow doesnt work on mac for some reason

    # cv2.rectangle(masked_image,(15,20),(70, 50), ( 0, 55, 255), 2)

    small = cv2.resize(masked_image, (0,0), fx=0.5, fy=0.5)
    small2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    small3 = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('frame',small)
    # cv2.imshow('frame2',small2)
    # cv2.imshow('frame3',small3)

    blur_factor = cv2.getTrackbarPos ('blur_factor', 'controls')
    hmin = cv2.getTrackbarPos ('Hmin', 'controls')
    hmax = cv2.getTrackbarPos ('Hmax', 'controls')
    smin = cv2.getTrackbarPos ('Smin', 'controls')
    smax = cv2.getTrackbarPos ('Smax', 'controls')
    vmin = cv2.getTrackbarPos ('Vmin', 'controls')
    vmax = cv2.getTrackbarPos ('Vmax', 'controls')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
