import numpy as np
import argparse
import imutils
import cv2


def sort_contours(cnts, method="left-to-right"):
    # Initalize reverse flag and sort index
    reverse = False
    i = 0

    # Check if need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # Check if sorting against y coordinate instead of x
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Construct list of bounding boxes, sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)


def draw_contour(image, c, i):
    # Compute center of contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    return image


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())

# Load image, initilze accumulated edge image
image = cv2.imread(args["image"])
accumEdged = np.zeros(image.shape[:2], dtype="uint8")

# Loop over blue, green, and red channels
for chan in cv2.split(image):
    # Blur channel, extract edges, and accumulate edge set
    chan = cv2.medianBlur(chan, 11)
    edged = cv2.Canny(chan, 50, 200)
    accumEdged = cv2.bitwise_or(accumEdged, edged)

cv2.imshow("Edge Map", accumEdged)
cv2.waitKey(0)

# Find contours in accumulated image
cnts = cv2.findContours(
    accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
orig = image.copy()

# Loop over unsorted containers
for (i, c) in enumerate(cnts):
    orig = draw_contour(orig, c, i)

# Show original, unsorted image
cv2.imshow("Unsorted", orig)

# Sort contours
(cnts, boudingBoxes) = sort_contours(cnts, method=args["method"])

# Draw sorted contours
for (i, c) in enumerate(cnts):
    draw_contour(image, c, i)

# Show result image
cv2.imshow("Sorted", image)
cv2.waitKey(0)
