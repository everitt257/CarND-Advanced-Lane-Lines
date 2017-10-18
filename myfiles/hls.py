# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:40:17 2017

@author: Xuandong Xu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    # thresh = (90,255)
    binary = np.zeros_like(S)
    binary[(S>thresh[0]) & (S<=thresh[1])] = 1
    return binary

def h_channel(img,thresh=(15,100)):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    binary = np.zeros_like(H)
    binary[(H>thresh[0]) & (H<=thresh[1])] = 1
    return binary
    
def color_thresh(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    yellow_min = np.array([15, 100, 120], np.uint8)
    yellow_max = np.array([80, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 30, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)

    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1
    
    filtered = img
    filtered[((yellow_mask == 0) & (white_mask == 0))] = 0
#    plt.imshow(filtered,cmap='hsv')

    return binary_output
    
    
if __name__ == "__main__":
    img = mpimg.imread("../test_images/test2.jpg")
    hls_binary = color_thresh(img)
    #hls_binary = hls_select(img,thresh=(170,255))
    #plt.imshow(hls_binary,cmap="gray")