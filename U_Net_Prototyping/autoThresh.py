import os
import natsort
import numpy as np
import cv2

path = 'U_Data/U_Train/images/'

done_path = 'U_Data/U_Train/masks/'

lst = os.listdir(path)
imlist = []

natsort.natsorted(lst,reverse=False)
i = 1

for file in os.listdir(path):
    imlist.append(file)

print(imlist)
imlist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in imlist:
   os.rename(os.path.join(path, file), os.path.join(path, str(i)+'_cam-image_array_'+ '.jpg'))
   i = i+1
imlist = []
for file in os.listdir(path):
    imlist.append(file)

imlist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

print(imlist)
print(i)

    
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold_white = [red_threshold, green_threshold, blue_threshold]

i = 1
for file in imlist:
    image = cv2.imread(path + str(i) + '_cam-image_array_.jpg', 1)
    colorMaskWhite = (image[:,:,0] < rgb_threshold_white[0]) | \
                    (image[:,:,1] < rgb_threshold_white[1]) | \
                    (image[:,:,2] < rgb_threshold_white[2])
    localImage = np.copy(image)
    localImage[colorMaskWhite] = [0,0,0]
    (height, width) = localImage.shape[:2]
    point = round(height/3)
    localImage[:point, :width, :] = [0,0,0]
    localImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(done_path + str(i) +'_mask_.jpg', localImage)
    i = i+1

cv2.imshow('threshold example', localImage)
print(i)