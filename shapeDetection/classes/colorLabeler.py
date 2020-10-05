import scipy.spatial import distance as dist
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
