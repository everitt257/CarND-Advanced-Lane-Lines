# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:45:25 2017

@author: everitt257
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip

from scipy import misc            
from undistort_img import undistort_img as undistorter
from threshhold_img import threshhold_img as thresher
from warp_img import warp_img
from line_test1 import line
from addtext import addtext_img

# define helper class for image processing
addtexter = addtext_img()
fitliner = line()
warper = warp_img()
thresh = thresher()
undist = undistorter()

# define pipline for image processing
def pipline(img):
    
    #testimg = mpimg.imread("../test_images/test2.jpg")
    img = undist.cal_undistort(img)
    oldimg = np.copy(img)
    #misc.imsave('./undistorted.jpg',img)
    img = thresh.sobel_color(img)
    #misc.imsave('./thresholdedbinary.jpg',img*255)
    img = warper.getwarped(img)
    #misc.imsave('./perspectivetransformed.jpg',img*255)
    img = fitliner.line_main(img,oldimg,warper.Minv)
         
    addtexter.addtext(img,fitliner.offset,fitliner.curvature,fitliner.detected)
    #misc.imsave('./added_effect.jpg',img)
    return img

## this is where the main program runs
if __name__ == "__main__":
    # video = 'harder_challenge_video'
    # video = 'challenge_video'
    video = '../project_video'
    white_output = '{}_rendered.mp4'.format(video)
    clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(10, 50)
    white_clip = clip1.fl_image(pipline)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
