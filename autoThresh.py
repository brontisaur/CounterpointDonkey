import os
import natsort
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'halldatarecords2/'

done_path = 'thresh_im_2/'

lst = os.listdir(path)
imlist = []
images = []
imagesGray=[]
medians =[]
finalImages=[]

for file in os.listdir(path):
    if 'cam' in file:
        imlist.append(file)

    
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold_white = [red_threshold, green_threshold, blue_threshold]
i=1
for file in imlist:
    image = cv2.imread(path + str(i) + '_cam-image_array_.jpg', 1)
    images.append(image)
    i=i+1

for i in range(0, len(images)):
    imgGray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    imagesGray.append(imgGray)
    median = np.median(imgGray)
    medians.append(median)

for n in range(0, len(images)):
    imgEdit = imagesGray[n]
    print(imgEdit.shape)
    colorMaskWhite = (imgEdit[:,:] < round(medians[n] + medians[n]*0.3))
    localImage = np.copy(imgEdit)
    localImage[colorMaskWhite] = 0
    (height, width) = localImage.shape[:2] 
    point = round(height/2.5)
    localImage[:point, :width] = 0
    finalImages.append(localImage)
    cv2.imwrite(done_path + str(n+1) +'_mask_.jpg', localImage)

print(colorMaskWhite)
dummy = cv2.imread(done_path +'1000_mask_.jpg',0)
print(dummy.shape)
plt.imshow(localImage)
plt.show()
#i = 1
#for file in imlist:
#    image = cv2.imread(path + str(i) + '_cam-image_array_.jpg', 1)
#    print(image.shape)
#    colorMaskWhite = (image[:,:,0] < rgb_threshold_white[0]) | \
#                    (image[:,:,1] < rgb_threshold_white[1]) | \
#                    (image[:,:,2] < rgb_threshold_white[2])
#    localImage = np.copy(image)
#    localImage[colorMaskWhite] = [0,0,0]
#    (height, width) = localImage.shape[:2]
#    point = round(height/3)
#    localImage[:point, :width, :] = [0,0,0]
    #localImage = cv2.cvtColor(localImage, cv2.COLOR_BGR2GRAY)
#    cv2.imwrite(done_path + str(i) +'_mask_.jpg', localImage)
#    i = i+1

#cv2.imshow('threshold example', localImage)
#print(i)