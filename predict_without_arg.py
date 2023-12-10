# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import imutils

# manually specify the input image path
imagePath = "/home/supreeta/Documents/HiWi/Objekterkennung/pyimage_code/dataset/images/nrfp/image_25.png"

# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)

# load the input image (in Keras format) from disk and preprocess
# it, scaling the pixel intensities to the range [0, 1]
image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# make bounding box predictions on the input image
preds = model.predict(image)[0]
(startX, startY, endX, endY) = preds

print("StartX:", startX)
print("StartY:", startY)
print("EndX:", endX)
print("EndY:", endY)


image = cv2.imread(imagePath)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
# scale the predicted bounding box coordinates based on the image
# dimensions
startX = int(startX * w)
startY = int(startY * h)
endX = int(endX * w)
endY = int(endY * h)

print("StartX:", startX)
print("StartY:", startY)
print("EndX:", endX)
print("EndY:", endY)

# # draw the predicted bounding box on the image
# cv2.rectangle(image, (startX, startY), (endX, endY),
# 	(0, 255, 0), 2)
# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)
