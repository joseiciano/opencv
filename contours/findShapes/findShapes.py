import numpy as np
from argparse import ArgumentParser
import imutils
import cv2

ap = ArgumentParser()
ap.add_argument("-i", "--image", help="path to image file")
args = vars(ap.parse_args())

# Load image
image = cv2.imread(args["image"])

# Find all the black shapes in the image
lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv2.inRange(image, lower, upper)

# Grab contours
cnts = cv2.findContours(
    shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(f"Found {len(cnts)} black shapes")
cv2.imshow("Mask", shapeMask)

# Loop over contours
for c in cnts:
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
