#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from skimage import segmentation, color, measure

arg = sys.argv[1]

img = cv2.imread(arg) #  "Data/CD36P-231-6-DAPI.tif"

def show_img(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    cells=img[:,:, 0]  # this line shows the tif image 
    ax.imshow(cells,cmap='gray')

show_img(img) # this even shows the .tiff image

# Slice the channel
img.shape
cells=img[:,:, 0]  
cells.shape

#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh.shape
plt.imshow(thresh)

# Morphological operations to remove small noise - Using Erosion and Dilation
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
# Remove the cell touching the edges
opening = segmentation.clear_border(opening)
# Find sure background
sure_bg = cv2.dilate(opening,kernel,iterations=10)

# Apply distance transform
distance_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

# Threshold the distance transform by starting at half it's max value
# Find sure foreground
ret2, sure_fg = cv2.threshold(distance_transform,0.5 * distance_transform.max(), 255, 0)

# Find Unknown region 
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Create a marker and label the regions inside. 
ret3, markers = cv2.connectedComponents(sure_fg)

# Add 15 to all labels so that sure background is not 0, but 15
markers = markers + 15

# Mark the region of unknown with zero
markers[unknown==255] = 0

# Watershed algorithm
markers = cv2.watershed(img,markers)

# Set boundaries in pink color 
img[markers == -1] = [0,255,255]  
plt.imshow(img)

img2 = color.label2rgb(markers, bg_label=0)
plt.imshow(img2)

# Start : Find the stats 
pixels_to_um=0.454
props = measure.regionprops_table(markers, cells,
properties=['label', 'area', 'equivalent_diameter', 'mean_intensity', 'solidity', 'orientation', 'perimeter'])
df = pd.DataFrame(props)

df = df[df['area'] > 50]
print(df.head())

df['area_sq_microns'] = df['area'] * (pixels_to_um**2)
df['equivalent_diameter_microns'] = df['equivalent_diameter'] * (pixels_to_um)
print(df.head())
df.to_excel('data_to_excel.xlsx', sheet_name='measurements')
# End : Find the stats

# Start : Label the image
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
# End : Label the image

