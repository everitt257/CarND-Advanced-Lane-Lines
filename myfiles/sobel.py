# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:52:26 2017

@author: Xuandong Xu
"""

import numpy as np
import cv2
import matplotlib.image as mpimg





# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255,sobel_kernel=3):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    assert orient == 'x' or orient == 'y',"choose x or y"
    if orient == 'x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else: 
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel>=thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = img #cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx**2+sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    # 5) Create a binary mask where mag thresholds are met
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag>=mag_thresh[0]) & (gradmag<=mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = img #cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



    
if __name__ == "__main__":
    # Read in an image and grayscale it
    image = mpimg.imread('signs_vehicles_xygrad.png')
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # Run the function
    grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=40, thresh_max=100,sobel_kernel=7)
    grad_binary2 = mag_thresh(image,sobel_kernel = 7,mag_thresh=(50,100))
    grad_binary3 = dir_threshold(image,sobel_kernel = 7,thresh=(np.pi/6,np.pi/3))
    
    final = np.zeros_like(grad_binary)
    final[((grad_binary == 1) & (grad_binary3 == 1)) & (grad_binary2 == 1)] = 1
    # Plot the result
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(image,cmap='gray')
#    ax1.set_title('Original Image', fontsize=50)
#    ax2.imshow(final, cmap='gray')
#    ax2.set_title('Thresholded Gradient', fontsize=50)
#    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    hist = np.sum(final[final.shape[0]/2:,:],axis = 0)
      