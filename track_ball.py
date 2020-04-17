# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# import the necessary packages
from collections import deque # maintain a list of the past N (x, y)-locations of the ball in our video stream. Maintaining such a queue allows us to draw the “contrail” of the ball as its being tracked.
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
				help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, 
				help="max buffer size") # length of path
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the 
# list of tracked points
# ball hsv = 7, 154, 86, https://imagecolorpicker.com/, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
ballColorLower = (5, 120, 65)#(29, 86, 6)
ballColorUpper = (9, 170, 105)#(64, 255, 255)
pts = deque(maxlen=args["buffer"])

video = 'videos/jordan3.mp4'
vs = cv2.VideoCapture(video)
hasFrame, frame = vs.read()
vid_writer = cv2.VideoWriter('videos/jordan3_track.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                             (frame.shape[1], frame.shape[0]))


# if a video path was not supplied, grab the reference
# to the webcam
#if not args.get("video", False):
#	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
#else:
#	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping until 'q' is pressed or video ends
while True:
	# grab the current frame
	hasFrame, frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=600) # process the frame faster, leading to an increase in FPS 
	blurred = cv2.GaussianBlur(frame, (11, 11), 0) # reduce high frequency noise and allow us to focus on the structural objects
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, ballColorLower, ballColorUpper) # handles the actual localization of the ball
	mask = cv2.erode(mask, None, iterations=2) # erode and dilate to remove small blobs
	mask = cv2.dilate(mask, None, iterations=2)
	cv2.imshow("mask",mask)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea) # largest contour
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		
	vid_writer.write(frame)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	cv2.waitKey()
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
#if not args.get("video", False):
#	vs.stop()

# otherwise, release the camera
#else:
#	vs.release()

vid_writer.release()
# close all windows
cv2.destroyAllWindows()