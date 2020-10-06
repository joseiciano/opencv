from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):
        # Initalize colors dictionary
        # Maps color name -> RGB tuple
        colors = OrderedDict({
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        })

        # Allocatte memory L*a*b* image
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # Loop over colors dict
        for (i, (name, rgb)) in enumerate(colors.items()):
            # Update L*a*b* array and color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

        # Convert L*a*b* array from RGB to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image, c):
        # Construct mask for contour
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)

        # Calcaulte average L*a*b* value for the masked region
        mean = cv2.mean(image, mask=mask)[:3]

        minDist = (np.inf, None)

        # Loop over known L*a*b* color values
        for i, row in enumerate(self.lab):
            # Compute distance between current value and mean of image
            d = dist.euclidean(row[0], mean)

            # If distance is smaller than current, we have new min
            if d < minDist[0]:
                minDist = (d, i)

        # Return name of the color with this smallest dist
        return self.colorNames[minDist[1]]
