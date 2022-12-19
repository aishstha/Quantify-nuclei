#!/usr/bin/env python3

import cv2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

arg = sys.argv[1]

img = cv2.imread(arg) #  "Data/CD36P-231-6-DAPI.tif"

def show_img(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    cells=img[:,:, 0]  # this line shows the tif image 
    ax.imshow(cells,cmap='gray')

# show_img(img)
# cv2.imwrite("image_actual1.jpg", img)
img.shape

# Slice the channel
cells=img[:,:, 0]  
cells.shape

# This makes image with two channel, previously it was 3 channel
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img.shape

#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh.shape
plt.imshow(thresh)
# cv2.imwrite("image_actual12.jpg", thresh)

# Morphological operations to remove small noise - opening
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# finding sure background
sure_bg = cv2.dilate(opening,kernel,iterations=10)

#applying distance transform
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret2, sure_fg =cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

# Unknown region 
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#Now we create a marker and label the regions inside. 
ret3, markers = cv2.connectedComponents(sure_fg)

#add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Watershed
markers = cv2.watershed(img,markers)

# color boundaries in yellow. 
img[markers == -1] = [0,255,255]  
img2 = color.label2rgb(markers, bg_label=0)
plt.imshow(img2)

#thresholding a color image, here keeping only the yellow in the image
th=cv2.inRange(img,(0,255,255),(0,255,255)).astype(np.uint8)

#inverting the image so components become 255 seperated by 0 borders.
th=cv2.bitwise_not(th)

#calling connectedComponentswithStats to get the size of each component
nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(th,connectivity=4)

#taking away the background
nb_comp-=1; sizes=sizes[0:,-1]; centroids=centroids[1:,:]

bins = list(range(np.amax(sizes)))

#plot distribution of your cell sizes.

numbers = sorted(sizes)


plt.hist(sizes,numbers)
cv2.imwrite("image_with_boder.jpg", img)


labels = np.unique(markers)
for label in labels:
    y, x = np.nonzero(markers == label)
    cx = int(np.mean(x))
    cy = int(np.mean(y))
    color = (255, 255, 255)
    img[markers == label] = np.random.randint(0, 255, size=3)
    cv2.circle(img, (cx, cy), 2, color=color, thickness=-1)
    cv2.putText(img, f"{label}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

cv2.imwrite("output.jpg", img)

# Contour
img.shape
cells=img[:,:, 0]  
cells.shape

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print ("Number of cell in image:" , len(contours))

def find_stats_and_export(contours, image):
    data = []
    df = pd.DataFrame()

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        perimeter = cv2.arcLength(contours[i], True)

        # calculate number of pixels: set an ROI around the contour and compute the area with CountNonZero applied on this area.
        cimg = np.zeros_like(image.copy())
        cv2.drawContours(cimg, contours, i, (255,255,255), thickness=cv2.FILLED) # 12th dot and 1 thick boder 
        pixel_count = np.count_nonzero(cimg)

        # if pixel_count > 6:
        data.append((area, pixel_count, perimeter))
       
 
    #load data into a DataFrame object:
    df = pd.DataFrame(data, columns=('Area', 'Pixel', 'Perimeter'))
    print(df)
    df.to_excel('data_to_excel.xlsx', sheet_name='cell_stats')

find_stats_and_export(contours, thresh)


