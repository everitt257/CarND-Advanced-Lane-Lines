# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:23:05 2017

@author: Xuandong XU
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class warp_img:
    def __init__(self):
        self.src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

        self.dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst,self.src)
        
    def getwarped(self,img):
        warped = cv2.warpPerspective(img, self.M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped
        
    def unwarped(self,img):
        unwarped = cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]))
        return unwarped

from undistort_img import undistort_img as undistorter
from threshhold_img import threshhold_img as thresher
        
if __name__ == "__main__":
    warper = warp_img()
    thresh = thresher()
    undist = undistorter()

    testimg = mpimg.imread("../test_images/straight_lines1.jpg")
    #pts = warper.src.reshape((-1,1,2))
    cv2.polylines(testimg,np.int32([warper.src]),True,(255,0,0))
    img = undist.cal_undistort(testimg)
    #img = thresh.sobel_color(img)
    img = warper.getwarped(img)
    cv2.polylines(img,np.int32([warper.dst]),True,(255,0,0))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(testimg)
    ax1.set_title('Original Img', fontsize=50)
    ax2.imshow(img)
    ax2.set_title('Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    