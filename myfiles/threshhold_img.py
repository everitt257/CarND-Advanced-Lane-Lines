# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 00:44:27 2017

@author: Xuandong Xu
"""
from sobel import abs_sobel_thresh,dir_threshold,mag_thresh
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from hls import hls_select,color_thresh

class threshhold_img:
    def __init__(self):
        pass
    
    def sobel_thresh(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #perform different sobel operation
        dirx_binary = abs_sobel_thresh(gray, orient='x', thresh_min=30, thresh_max=100,sobel_kernel=7)
        mag_binary = mag_thresh(gray,sobel_kernel = 7,mag_thresh=(30,100))
        tan_binary = dir_threshold(gray,sobel_kernel = 7,thresh=(np.pi/6,np.pi/3))
        # sobelimg combination
        newimg = np.zeros_like(dirx_binary)
        newimg[(dirx_binary == 1) | ((mag_binary == 1) & (tan_binary == 1))] = 1
        #return
        return newimg
        
    def color_thresh(self,img):
        #newimg = hls_select(img, thresh=(170, 255))
        newimg = color_thresh(img)
        return newimg
    
    def sobel_color(self,img):
        sobelimg = self.sobel_thresh(img)
        colorimg = self.color_thresh(img)
        newimg = np.zeros_like(sobelimg)
        newimg[(sobelimg == 1) | (colorimg == 1)] = 1
        return newimg

from undistort_img import undistort_img as undistorter
        
if __name__ == "__main__":
    tresh = threshhold_img()
    undist = undistorter()
    testimg = mpimg.imread("../test_images/test2.jpg")
    img = undist.cal_undistort(testimg)
    img = tresh.sobel_color(img)
    
    
