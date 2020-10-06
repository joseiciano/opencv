import numpy as np
import cv2


def orderPoints(pts):
    # Initialize list of coordinates that will be ordered
    # [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left point has smallest sum
    # Bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute difference between points
    # Top-right has smallest difference
    # Bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return ordered coordinates
    return rect


def distformula(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return np.sqrt((x**2) + (y**2))


def fourPointTransform(image, pts):
    # Obtain consistent order of points, unpack individually
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    # Width is max dist between bottom-right and bottom-left x-coords
    # Comparing to dist between top-right and top-left x-coords
    widthA = distformula(br, bl)
    widthB = distformula(tr, tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image
    # Height is max dist between top-right and bottom-right y-coords
    # Compared to dist between top-left and bottom-left y-coords
    heightA = distformula(tr, br)
    heightB = distformula(tl, bl)
    maxHeight = max(int(heightA), int(heightB))

    # Create dimensions of new image
    # Construct set of destination points to generate birds eye view
    # Order of points: [top-left, top-right, bottom-right, bottom-left]
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Perspective matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
