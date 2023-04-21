import cv2
import numpy as np

# Load the image
image = cv2.imread('red-round-glossy-button-isolated.png')

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range of red color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image, image, mask=mask)

# Display the resulting frame
cv2.imshow("mask", mask)
cv2.imshow('frame', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
