import numpy as np
import argparse
import cv2

# Args parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())

# Load image
image = cv2.imread(args["image"])

# Color boundaries, (color scheme: BGR)
# Each row shows the (lower, upper) limit for categorizing a color
boundaries = [
    ([17, 15, 100], [50, 56, 200]),  # Red
    ([86, 31, 4], [220, 88, 50]),  # Blue
    ([25, 146, 190], [62, 174, 250]),  # Yellow
    ([103, 86, 65], [145, 133, 128])  # Gray
]

for (lower, upper) in boundaries:
    # Create numpy arrays from boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Find the colors within the specific boundaries
    # Generate a mask of that color boundary
    mask = cv2.inRange(image, lower, upper)

    # Apply mask to the original image
    output = cv2.bitwise_and(image, image, mask=mask)

    # Show the original image, and the new output side by side
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)
