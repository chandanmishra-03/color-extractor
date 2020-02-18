import cv2
import numpy as np

from color_extractor import ImageToColor

npz = np.load('color_names.npz')
img_to_color = ImageToColor(npz['samples'], npz['labels'])

img = cv2.imread('image.jpg')
print(img_to_color.get(img))
