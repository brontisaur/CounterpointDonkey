import cv2
import numpy as np

import matplotlib.pyplot as plt

images = []

image = cv2.imread('101_cam-image_array_.jpg')

images.append(image)

image = cv2.imread('lanes.jpg')

images.append(image)

image = cv2.imread('arrowcity.jpg')

images.append(image)

imagesGray = []
medians = []
for i in range(0, len(images)):
    imgGray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    imagesGray.append(imgGray)
    median = np.median(imagesGray[i])
    medians.append(median)


print(medians)

finalImages = []
#Threshold grayscaled image according to median:
for n in range(0, len(images)):
    imgEdit = imagesGray[n]
    print(n)
    colorMaskWhite = (imgEdit[:,:] < round(median + median*0.3))
    print(colorMaskWhite.shape)
    localImage = np.copy(imgEdit)
    localImage[colorMaskWhite] = 0
    (height, width) = localImage.shape[:2] 
    point = round(height/2.5)
    localImage[:point, :width] = 0
    finalImages.append(localImage)

#Visualise a comparison
plt.imshow(np.squeeze(finalImages[2]))
plt.show()
plt.imshow(np.squeeze(images[2]), cmap='gray')
plt.show()