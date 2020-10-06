import numpy as np
import cv2
import argparse


def show_image(title, image, width=300):
    # Resize the image to have constant width
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Show resized image
    cv2.imshow(title, resized)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def image_stats(image):
    # Computs mean and standard deviation of each color channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    return (lMean, lStd, aMean, aStd, bMean, bStd)


def color_transfer(source, target):
    # Convert images from RGB to L*ab*
    # Use float32 because opencv expects floats as 32bit, not 64
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Compute color statistics
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # Substract means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # Scale by standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # Add in source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # Clip the pxel intensitives to [0, 255]
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Merge channels together, converting back to rgb
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    return transfer


# Arguments parser
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to source image")
ap.add_argument("-t", "--target", required=True, help="Path to target image")
args = vars(ap.parse_args())

# Load image
source = cv2.imread(args["source"])
target = cv2.imread(args["target"])

# Transfer the color distribution from the source image to the target
transfer = color_transfer(
    source, target)

show_image("Source", source)
show_image("Target", target)
show_image("Transfer", transfer)
cv2.waitKey(0)
