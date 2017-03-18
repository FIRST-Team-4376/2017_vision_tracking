import cv2
from networktables import NetworkTables

cap = cv2.VideoCapture(0)
ret, img = cap.read()
# small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
# cv2.imshow('controls',small)

NetworkTables.initialize(server='roborio-4376-frc.local');
sd = NetworkTables.getTable("SmartDashboard")

hmin = 58
hmax = 94

smin = 44
smax = 255

vmin = 20
vmax = 255

hsv_min = np.array([hmin,smin,vmin])
hsv_max = np.array([hmax,smax,vmax])

while(True):
  ret, img = cap.read()
  hsv_min = np.array([hmin,smin,vmin])
  hsv_max = np.array([hmax,smax,vmax])
  ret, img = cap.read()

  (h, w) = img.shape[:2]
  center = (w / 2, h / 2)
  M2 = cv2.getRotationMatrix2D(center, 180, 1.0)
  img = cv2.warpAffine(img, M2, (w, h))

  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_img, hsv_min, hsv_max)

  masked_image = cv2.bitwise_and(img,img, mask= mask)