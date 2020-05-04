# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# import the necessary packages
from collections import deque # maintain a list of the past N (x, y)-locations of the ball in our video stream. Maintaining such a queue allows us to draw the “contrail” of the ball as its being tracked.
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from scipy.optimize import linear_sum_assignment

# define the lower and upper boundaries of the "green" ball in the HSV color space, then initialize the list of tracked points
# ball_steph2: (10, 174, 138), low = (8, 160, 130), high = (12, 180, 150)
# ball_jordan3 = (7, 154, 86) low = (6,145,75), high = (9, 165, 95) https://imagecolorpicker.com/, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
ballColorLower = (8,150,100)#(8,150,110) #(3,100,110) # (3,140,75)#(5,140,75)
ballColorUpper = (14,255,255)#(14,255,255)#(10, 170, 95)
lower_area = 20
upper_area = 200
dist = 150
frames = 40
trace = 3500

pts = deque(maxlen=64)

play_name = 'fist21'
video = 'videos/2ktest.mp4'#spurs_play_'+play_name+'.mp4'
vs = cv2.VideoCapture(video)
hasFrame, frame = vs.read()
#vid_writer = cv2.VideoWriter('videos/2kpc_test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))

# allow the camera or video file to warm up
time.sleep(2.0)


"""uses code from https://github.com/srianant/kalman_filter_multi_object_tracking"""
class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]
            #print(self.tracks[i].prediction)
            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = self.lastResult
        else:  # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                              self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u



tracker = Tracker(dist, frames, trace, 1)

count=0 #for counting the frames that have passed since the last pass
counter=0 #for counting the pts in the queue
passcount=0 #for counting passes
startCount=30 #for accounting for the first 30 frames
# keep looping until 'q' is pressed or video ends
while True:
	# grab the current frame
	hasFrame, frame = vs.read()
	possiblePass = False
	# reached the end of the video
	if frame is None:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Blur using 3 * 3 kernel.
	gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0) # cv2.blur(gray, (3, 3))

	# resize the frame, blur it, and convert it to the HSV color space
	#frame = imutils.resize(frame, width=1000) # process the frame faster, leading to an increase in FPS
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

	cntr = ([[-1], [-1]])
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid

		# detect near-circles in the hsv filtered image
		contour_list = []
		circularity_list = []
		for contour in cnts:
			epsilon = 0.2*cv2.arcLength(contour,True)
			approx = cv2.approxPolyDP(contour,epsilon,True)
			area = cv2.contourArea(contour)
			# Filter based on length and area
			if (1 < len(approx) < 1000) & (upper_area > area > lower_area):

				contour_list.append(contour)
				area = cv2.contourArea(contour)
				perimeter = cv2.arcLength(contour,True)
				circularity = (4*np.pi * area) / (perimeter**2)
				circularity_list.append(circularity)

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
			#vid_writer.write(frame)
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
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		cntr = ([[cX], [cY]])

		# only proceed if the radius meets a minimum size
		#if radius > 10:
			#if c in contour_list:
			# draw the circle and centroid on the frame, then update the list of tracked points
		cv2.circle(frame, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
		cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)
	centers = []
	counter= counter+1
	if(cntr != ([[-1], [-1]])):
		centers.append(cntr)

	tracker.Update(centers)
	if (len(pts) > 0):
		for i in range(len(tracker.tracks)):
			if (True):
				for j in range(len(tracker.tracks[i].trace)-2):
					# Draw trace line
					if(i==0):
						x1 = tracker.tracks[i].trace[j][0][0]     #get the last 3 points of movement
						y1 = tracker.tracks[i].trace[j][1][0]
						x2 = tracker.tracks[i].trace[j+1][0][0]
						y2 = tracker.tracks[i].trace[j+1][1][0]
						x3 = tracker.tracks[i].trace[j+2][0][0]
						y3 = tracker.tracks[i].trace[j+2][1][0]
						thickness = int(np.sqrt(64 / float(i + 1)) )

						a = np.array([x1,y1]) #organize as points
						b = np.array([x2,y2])
						c = np.array([x3,y3])

						ba = a - b #for angle calculation
						bc = c - b

						cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) #get cosine angle
						angle = np.arccos(cosine_angle) #get the actual angle in radians
						angleD = np.degrees(angle) #convert to degrees
						if((angleD < 60 and angleD > 0) or (angleD < 360 and angleD > 300)): #if the angle b/t the last 3 points is <60 degrees
							if(count>35 or startCount>0): #if enough frames have passed since last pass was detected (startCount accounts for the beginning 30 frames)
								print("POSSIBLE PASS MADE")
								possiblePass = True     #label as possibly a pass
								count = 0               #reset counter to 0
    #detect direction of balls path (https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/)
	for i in np.arange(1, len(tracker.tracks[0].trace)):
		# if either of the tracked points are None, ignore
		# them
		if tracker.tracks[0].trace[i - 1] is None or tracker.tracks[0].trace[i] is None:
			continue
		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 1 and tracker.tracks[0].trace[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = tracker.tracks[0].trace[-10][0] - tracker.tracks[0].trace[i][0] #get the average x and y movement of ball in the past 10 frames
			dY = tracker.tracks[0].trace[-10][1] - tracker.tracks[0].trace[i][1]
			# ensure there is significant movement in the
			# x-direction
			#if(dY>dX):
			#	possiblePass = False

			if np.abs(dX) > 50:
				dirX = "East" if np.sign(dX) == 1 else "West"
				#print(dirX)
			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 120:
				dirY = "North" if np.sign(dY) == 1 else "South"
				#print(dirY)
				if(possiblePass==True):
					possiblePass = False #if the path of the ball in the last 10 frames is >120 pixels of vertical movement, do not count as a pass
					#print("too vertical")



	if(possiblePass==True):
		passcount = passcount+1 #update the pass count
	count=count+1 #incrememnt for counting frames between passes
	#print on the pass count on screen
	cv2.putText(frame, "PASSES MADE: " + str(passcount), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	#vid_writer.write(frame) #writes to a video

	cv2.imshow("Frame", frame)
	cv2.waitKey()
	#key = cv2.waitKey(1) & 0xFF
	startCount = startCount-1
	# if the 'q' key is pressed, stop the loop
	#if key == ord("q"):
	#	break

#vid_writer.release()
cv2.destroyAllWindows()
