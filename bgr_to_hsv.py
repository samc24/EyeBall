import numpy as np
import cv2

color = np.uint8([[[39,77,139]]])
hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
print (hsv_color)