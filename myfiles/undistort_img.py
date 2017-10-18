# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:45:31 2017

@author: Xuandong Xu
"""

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

class undistort_img:
    def __init__(self):
        try:
            favoritepoints = pickle.load(open( "savedpoints.p", "rb" ))
            self.objpoints = favoritepoints["objpoints"]
            self.imgpoints = favoritepoints["imgpoints"]
            #print('pickle file found')
        except:
            favoritepoints = None
            #print('pickle file not found')
        
        if favoritepoints is None:
            self.findcorners()
        
    def findcorners(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.
        
        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')
        
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        
        mypoints = {"objpoints":self.objpoints, "imgpoints":self.imgpoints}
        pickle.dump(mypoints, open("savedpoints.p", "wb"))
        
    def cal_undistort(self,img):
        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(self.objpoints,self.imgpoints,img.shape[0:2],None,None)
        undist = cv2.undistort(img,mtx,dist,None,mtx)
        return undist
        
if __name__ == "__main__":
    temp = undistort_img()
    img = mpimg.imread('../test_images/test3.jpg')
    newimg = temp.cal_undistort(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Img', fontsize=50)
    ax2.imshow(newimg)
    ax2.set_title('Corrected Img', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        
    
