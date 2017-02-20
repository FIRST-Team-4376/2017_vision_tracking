import numpy as np
import cv2
from networktables import NetworkTables

# rgb_min_blue = np.uint8([[[27, 60, 40]]])
# rgb_max_blue = np.uint8([[[81,255,120 ]]])
# hsv_max = cv2.cvtColor(rgb_max_blue,cv2.COLOR_BGR2HSV)
# hsv_min = cv2.cvtColor(rgb_min_blue,cv2.COLOR_BGR2HSV)

# green tape

def nothing(x):
	pass


def find_closest_value(target_value, list_of_values):
	closest_value = None
	for val in list_of_values:
		if closest_value is None:
			closest_value = val
		else:
			new_difference = abs(val - target_value)
			if new_difference < abs(closest_value - target_value):
				closest_value = val
	return closest_value


def width_and_height_from_contour(contour, approximation_value):
	epsilon = approximation_value * cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, epsilon, True)
	x,y,w,h = cv2.boundingRect(approx)
	return [float(w), float(h)]

def bounding_rectangle_coords_for_contour(contour, approximation_value):
	epsilon = approximation_value * cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, epsilon, True)
	x,y,w,h = cv2.boundingRect(approx)
	return {'x': x, 'y': y, 'width': w, 'height': h}

def width_height_ratio_pct_difference(width, height):
	target_ratio = (2.0 / 5.0)
	actual_ratio = (float(width) / float(height))
	result = abs( (actual_ratio - target_ratio) / target_ratio )
	if result > 1.0:
		return .999999999
	else:
		return result

def top_edge_same_height_score(rect_coords_to_score, bounding_rectangles_to_check_against, image_height):
	if len(bounding_rectangles_to_check_against) < 2:
		return 0.0
	else:
		removed_self_from_calculations = False
		top_edge_height_differences = []
		for rect_coords in bounding_rectangles_to_check_against:
			if rect_coords == rect_coords_to_score and removed_self_from_calculations == False:
				print "removing self!"
				removed_self_from_calculations = True
			else:
				top_edge_height_differences.append( abs(rect_coords_to_score['y'] - rect_coords['y']) )

		closest_value = find_closest_value(0.0, top_edge_height_differences)
		pct = float(closest_value) / float(image_height)
		return 1.0 - pct




def draw_bounding_rectangle(image_to_draw_on, contours, approximation_value):
	contour_centers = []
	# height, width = img.shape[:2]
	image_height, image_width = image_to_draw_on.shape[:2]

	differences_with_contours = []
	rects_with_scores = []
	bounding_rectangles_and_contours = []
	for found_contour in contours:
		bounding_rectangles_and_contours.append([bounding_rectangle_coords_for_contour(found_contour, approximation_value), found_contour])

	cv2.drawContours(image_to_draw_on, contours, -1, (0,255,0), 4)

	all_bounding_rects = []
	for thing in bounding_rectangles_and_contours:
		all_bounding_rects.append(thing[0])

	for bounding_rect_and_contour in bounding_rectangles_and_contours:

		bounding_rect_coords = bounding_rect_and_contour[0]
		contour = bounding_rect_and_contour[1]
		x = bounding_rect_coords['x']
		y = bounding_rect_coords['y']
		width = bounding_rect_coords['width']
		height = bounding_rect_coords['height']

		if width < height:
			width_height_ratio_score = 1.0 - width_height_ratio_pct_difference(width, height)
			top_edge_score = top_edge_same_height_score(bounding_rect_coords, all_bounding_rects, image_height)
			total_score = width_height_ratio_score + top_edge_score
			rects_with_scores.append([bounding_rect_coords, total_score, width_height_ratio_score, top_edge_score, contour])

	rects_with_scores = sorted(rects_with_scores, key=lambda x: x[1], reverse=True)

	for rect_and_score in rects_with_scores[:2]:  # [:2] gets the first 2
		rect = rect_and_score[0]
		score = rect_and_score[1]
		width_height_ratio_score = rect_and_score[2]
		top_edge_score = rect_and_score[3]
		contour = rect_and_score[4]

		M = cv2.moments(contour)
		if M["m00"] > 0:
			cX = int(M["m10"] / M["m00"]) # center x coord
			cY = int(M["m01"] / M["m00"]) # center y coord
			cv2.circle(image_to_draw_on, (cX, cY), 7, (255, 0, 255), -1)
			contour_centers.append([cX, cY])
			# if 0 < cX < image_height and 0 < cY < image_width:
			# 	contour_centers.append([cX, cY])
			# 	print "THING"
			# 	hsv = cv2.cvtColor(image_to_draw_on, cv2.COLOR_BGR2HSV)
			# 	print hsv[cX][cY]
			# 	cv2.putText(image_to_draw_on, repr(hsv[cX][cY]), (cX+5, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),2)

		if score == rects_with_scores[0][1]:
			cv2.rectangle(image_to_draw_on,(rect['x'],rect['y']),(rect['x']+rect['width'], rect['y']+rect['height']), (255,0,0), 4)
		else:
			cv2.rectangle(image_to_draw_on,(rect['x'],rect['y']),(rect['x']+rect['width'], rect['y']+rect['height']), (0,0,255), 4)

		cv2.putText(image_to_draw_on, repr(score), (rect['x']+rect['width']+5, rect['y']), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
		cv2.putText(image_to_draw_on, repr(width_height_ratio_score), (rect['x']+rect['width']+5, rect['y']+50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
		cv2.putText(image_to_draw_on, repr(top_edge_score), (rect['x']+rect['width']+5, rect['y']+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)





	if len(contour_centers) == 2:
		left_center_x = contour_centers[0][0]
		left_center_y = contour_centers[0][1]
		right_center_x = contour_centers[1][0]
		right_center_y = contour_centers[1][1]

		overall_mid_x = (left_center_x + right_center_x) / 2
		overall_mid_y = (left_center_y + right_center_y) / 2
		cv2.circle(image_to_draw_on, (int(overall_mid_x), int(overall_mid_y)), 7, (255, 0, 255), -1)

		sd.putNumber('leftCenterX', left_center_x)
		sd.putNumber('leftCenterY', left_center_y)
		sd.putNumber('rightCenterX', left_center_x)
		sd.putNumber('rightCenterY', left_center_y)
		sd.putNumber('overallCenterX', overall_mid_x)
		sd.putNumber('overallCenterY', overall_mid_y)


hmin = 60
hmax = 90

smin = 0
smax = 140

vmin = 240
vmax = 255

blur_factor = 40

approx_value_divisor = 2
approx_value = 1

cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar('blur_factor','controls', blur_factor, 50, nothing)
cv2.createTrackbar('Hmin','controls', hmin, 255, nothing)
cv2.createTrackbar('Hmax','controls', hmax, 255, nothing)
cv2.createTrackbar('Smin','controls', smin, 255, nothing)
cv2.createTrackbar('Smax','controls', smax, 255, nothing)
cv2.createTrackbar('Vmin','controls', vmin, 255, nothing)
cv2.createTrackbar('Vmax','controls', vmax, 255, nothing)
cv2.createTrackbar('approx_value_divisor','controls', approx_value_divisor, 10, nothing)
cv2.createTrackbar('approx_value','controls', approx_value, 255, nothing)


cap = cv2.VideoCapture(0)
ret, img = cap.read()
small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.imshow('controls',small)

NetworkTables.initialize(server='roborio-4376-frc.local');
sd = NetworkTables.getTable("SmartDashboard")

while(True):

	hsv_min = np.array([hmin,smin,vmin])
	hsv_max = np.array([hmax,smax,vmax])
	ret, img = cap.read()
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
	while (blur_factor % 2 != 1):
		blur_factor += 1
	mask = cv2.medianBlur(mask,blur_factor)
	greyscale_image = mask

	masked_image = cv2.bitwise_and(img,img, mask= mask)

	ret2, thresh = cv2.threshold(mask, 150,255,cv2.THRESH_BINARY)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(masked_image, contours, -1, (0,255,0), 3)
	final_approx_value = float(approx_value) / pow(10.0, approx_value_divisor)
	print "approx_value"
	print approx_value
	print "approx_value_divisor"
	print approx_value_divisor
	print "final_approx_value"
	print final_approx_value
	draw_bounding_rectangle(img, contours, final_approx_value)


	# imshow doesnt work on mac for some reason

	# cv2.rectangle(masked_image,(15,20),(70, 50), ( 0, 55, 255), 2)

	small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	#small2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	#small3 = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
	small_thresh = cv2.resize(thresh, (0,0), fx=0.5, fy=0.5)
	cv2.imshow('frame',small)
	# cv2.imshow('frame2',small2)
	cv2.imshow('frame3',small_thresh)

	blur_factor = cv2.getTrackbarPos ('blur_factor', 'controls')
	hmin = cv2.getTrackbarPos ('Hmin', 'controls')
	hmax = cv2.getTrackbarPos ('Hmax', 'controls')
	smin = cv2.getTrackbarPos ('Smin', 'controls')
	smax = cv2.getTrackbarPos ('Smax', 'controls')
	vmin = cv2.getTrackbarPos ('Vmin', 'controls')
	vmax = cv2.getTrackbarPos ('Vmax', 'controls')
	approx_value_divisor = cv2.getTrackbarPos ('approx_value_divisor', 'controls')
	approx_value = cv2.getTrackbarPos ('approx_value', 'controls')

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
