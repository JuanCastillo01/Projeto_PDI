import numpy as np
import os
import cv2

def segmentImage(quantifiedImage):
    gray = cv2.cvtColor(quantifiedImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    # sure background area
    sure_bg = cv2.dilate(thresh, kernel, 5)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(quantifiedImage, markers)
    quantifiedImage[markers == -1] = [255, 255, 255]

    return quantifiedImage

# Lista de arquivos na pasta
samples = os.listdir("Quantified Photos")

# Opera todos os arquivos na pasta
for path in samples:

    quantifiedImg = cv2.imread(f"Quantified Photos/{path}")

    segmentedImage = segmentImage(quantifiedImg)

    cv2.imshow(path, segmentedImage)
    cv2.waitKey(0)

    cv2.imwrite(f"Quantified Photos/{path}", segmentedImage)
