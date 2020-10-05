import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # Initalize shape name, approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # If shape is triangle, should have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # If shape has 4 vertices, could be square or rectangle
        elif len(approx) == 4:
            # Compute bounding box
            (x, y, w, h) = cv2.boundingRect(approx)

            # Compute aspect ratio
            ar = w / float(h)

            # A square has aspect ratio is equal to one
            if ar >= 0.95 and ar <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"

        # If shape is pentagon, should have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # Otherwise, assume shape is a circle
        else:
            shape = "circle"

        return shape
