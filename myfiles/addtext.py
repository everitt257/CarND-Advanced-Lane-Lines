# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:53:16 2017

@author: everitt257
"""

import cv2
import matplotlib.image as mpimg
import numpy as np

class addtext_img:
    
    def __init__(self):
        self.preoffset = None
        self.precurve = None
        self.lastdetect= False
        
    def addtext(self,img,offset,curve,detected):
        self.preoffset = offset
        self.precurve = curve
        self.lastdetect = detected
        font = cv2.FONT_HERSHEY_SIMPLEX
        linetype = cv2.LINE_AA
        cv2.putText(img,'The weighted curvature is: {}m'.format(round(curve,2)),(20,50),font,1,(255,255,0),2,linetype)
        cv2.putText(img,'The car is {}m from the center'.format(round(offset,2)),(20,100),font,1,(255,255,0),2,linetype)
    
from scipy import misc            
from undistort_img import undistort_img as undistorter
from threshhold_img import threshhold_img as thresher
from warp_img import warp_img
from line_test1 import line
        
if __name__ == "__main__":
    addtexter = addtext_img()
    fitliner = line()
    warper = warp_img()
    thresh = thresher()
    undist = undistorter()
    

    testimg = mpimg.imread("../test_images/test2.jpg")
    img = undist.cal_undistort(testimg)
    misc.imsave('./undistorted.jpg',img)
    oldimg = np.copy(img)
    img = thresh.sobel_color(img)
    misc.imsave('./thresholdedbinary.jpg',img*255)
    img = warper.getwarped(img)
    misc.imsave('./perspectivetransformed.jpg',img*255)
    img = fitliner.line_main(img,oldimg,warper.Minv)
         
    addtexter.addtext(img,fitliner.offset,fitliner.curvature,fitliner.detected)
    #misc.imsave('./added_effect.jpg',img)
    misc.imsave('./added_effect.jpg',img)
    
    
    
    
    