from helpers.transform import fourPointTransform
import numpy as np
import argparse
import cv2

# Arguments parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords",
                help="comma seperated list of source points")
args = vars(ap.parse_args())

# Loads the image and grabs the source coordinates (i.e. list of (x, y))
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

# Apply four points transforms to obtain birds eye view
warped = fourPointTransform(image, pts)

# Show original and warped
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
