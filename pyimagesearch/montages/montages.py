from imutils import build_montages
from imutils import paths
import argparse
import random
import cv2

# Arguments parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-s", "--sample", type=int, default=21,
                help="# of images to sample")
args = vars(ap.parse_args())

# Grab paths to the images
imagePaths = list(paths.list_images(args["images"]))

# Randomly get a sample of images
random.shuffle(imagePaths)
imagePaths = imagePaths[:args["sample"]]

# Initalize list of images
images = []

# Loop over list of image paths
# For each image, load it and append it to images list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# Construct montages for the images
montages = build_montages(images, (128, 196), (7, 3))

# Loop over the montages, showing each
for montage in montages:
    cv2.imshow("Montage", montage)
    cv2.waitKey(0)
