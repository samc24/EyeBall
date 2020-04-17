import time
import cv2 
import numpy as np 
  
# Read image. 
img = cv2.imread('templates/jordan_3.png', cv2.IMREAD_COLOR) 
 
video = 'videos/jordan3.mp4'
cap = cv2.VideoCapture(video)
hasFrame, frame = cap.read()
vid_writer = cv2.VideoWriter('videos/jordan3_detect.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))

# allow the camera or video file to warm up
time.sleep(2.0)

while True:
    hasFrame, img = cap.read()
    if img is None:
        cv2.waitKey()
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
  
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  cv2.HOUGH_GRADIENT, 1, 100, param1 = 300, param2 = 30, minRadius = 30, maxRadius = 40) 

    # Draw circles that are detected. 
    if detected_circles is not None: 
  
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
  
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
  
            # Draw the circumference of the circle. 
            cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
  
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
    vid_writer.write(img)
    cv2.imshow("Detected Circle", img) 
    cv2.waitKey(0)

vid_writer.release()
# close all windows
cv2.destroyAllWindows()