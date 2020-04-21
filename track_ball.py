# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# import the necessary packages
from collections import deque # maintain a list of the past N (x, y)-locations of the ball in our video stream. Maintaining such a queue allows us to draw the “contrail” of the ball as its being tracked.
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

# define the lower and upper boundaries of the "green" ball in the HSV color space, then initialize the list of tracked points
# ball_steph2: (10, 174, 138), low = (8, 160, 130), high = (12, 180, 150)
# ball_jordan3 = (7, 154, 86) low = (6,145,75), high = (9, 165, 95) https://imagecolorpicker.com/, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
ballColorLower = (7,140,100)# (3,140,75)#(5,140,75)
ballColorUpper = (14,255,255)#(10, 170, 95) 
pts = deque(maxlen=64)

video = 'videos/spurs_play_fist21.mp4'
vs = cv2.VideoCapture(video)
hasFrame, frame = vs.read()
vid_writer = cv2.VideoWriter('videos/spurs_play_fist21_track.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping until 'q' is pressed or video ends
while True:
	# grab the current frame
	hasFrame, frame = vs.read()

	# reached the end of the video
	if frame is None:
		break
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	# Blur using 3 * 3 kernel. 
	gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0) # cv2.blur(gray, (3, 3)) 

	# resize the frame, blur it, and convert it to the HSV color space
	#frame = imutils.resize(frame, width=600) # process the frame faster, leading to an increase in FPS 
	blurred = cv2.GaussianBlur(frame, (11, 11), 0) # reduce high frequency noise and allow us to focus on the structural objects
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color, then perform a series of dilations and erosions to remove any small blobs left in the mask
	
	element = None #np.ones((5,5)).astype(np.uint8) # element for morphology, default None
	mask = cv2.inRange(hsv, ballColorLower, ballColorUpper) # handles the actual localization of the ball
	mask = cv2.erode(mask, element, iterations=2) # erode and dilate to remove small blobs
	mask = cv2.dilate(mask, element, iterations=2)
	cv2.imshow("mask",mask)

	# find contours in the mask and initialize the current (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
		
		# detect near-circles in the hsv filtered image
		contour_list = []
		circularity_list = []
		for contour in cnts:
			epsilon = 0.2*cv2.arcLength(contour,True) # jordan3: 0.05
			approx = cv2.approxPolyDP(contour,epsilon,True)
			area = cv2.contourArea(contour)
			# Filter based on length and area
			if (1 < len(approx) < 1000) & (450 >area > 130): # jordan3: (1 < len(approx) < 1000) & (5000 >area > 1000): 
				#print("***")
				#print("epsilon", epsilon)
				#print("approx", approx)
				#print("area", area)
				contour_list.append(contour)
				area = cv2.contourArea(contour)
				perimeter = cv2.arcLength(contour,True)
				circularity = (4*np.pi * area) / (perimeter**2)
				#print("circularity", circularity)
				circularity_list.append(circularity)
				#print("***")
		#print() 
		#print()
		cv2.drawContours(frame, contour_list,  -1, (255,0,0), 2)
		
		# if picking based on circularity 
		if contour_list != []:
			most_circ = circularity_list[0]
			c = contour_list[0]
			for i in range(len(circularity_list)):
				if circularity_list[i] > most_circ:
					most_circ=circularity_list[i]
					c = contour_list[i]
		else:
			vid_writer.write(frame)
			# show the frame to our screen
			cv2.imshow("Frame", frame)
			cv2.waitKey()
			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break
			continue

		# if picking based on contour area
		#c = max(cnts, key=cv2.contourArea) # largest contour 

		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			#if c in contour_list:
			# draw the circle and centroid on the frame, then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and draw the connecting lines
		thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	vid_writer.write(frame)

	cv2.imshow("Frame", frame)
	cv2.waitKey()
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

vid_writer.release()
cv2.destroyAllWindows()