# http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
from __future__ import print_function
import cv2


cap = cv2.VideoCapture(1)
while(True):
    # load the image and convert it to grayscale
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # show original image
    # cv2.imshow("Original", img)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps, descs) = detector.detectAndCompute(gray, None)
    if descs is None:
        print("No descs!")
    else:
        print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))

    # draw the keypoints and show the output image
    cv2.drawKeypoints(img, kps, img, (0, 255, 0))
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
