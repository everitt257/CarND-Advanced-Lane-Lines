# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:51:00 2017

@author: everitt257
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class line:
    def __init__(self):
         # window settings
        self.window_width = 60 
        self.window_height = 80 # Break image into 9 vertical layers since image height is 720
        self.margin = 30 # How much to slide left and right for searching
        
        # parameter for deciding using look-ahead filter
        self.detected = False # was the line detected in last iteration
        
        # recent fits setting 
        self.recent4fitx_left = []
        self.recent4fitx_right = []
        self.prefitx_left = None
        self.prefitx_right = None
        self.count = 0
        self.maxcount = 4
        
        # current fit setting
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        
        # information to show on img
        self.offset = None
        self.curvature = None
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700
        
        # left and right points on img
        self.leftpoints = None
        self.rightpoints = None
    
    def update_recent_fit(self):
        if self.count < self.maxcount:
            self.recent4fitx_left.append(self.left_fitx)
            self.recent4fitx_right.append(self.right_fitx)
            self.count = self.count + 1
            #print('recent fit is less than 4')
        else:
            index = (self.count) % (self.maxcount)
            self.recent4fitx_left[index] = self.left_fitx
            self.recent4fitx_right[index] = self.right_fitx
            self.count = self.count + 1
            #print('recent fit is of size:',self.count)
        
    def average_fit(self):
        getave = lambda x:np.average(np.array(x),axis=0)
        if len(self.recent4fitx_left)>0:
            self.prefitx_left = self.left_fitx
            self.prefitx_right = self.right_fitx
            self.left_fitx = getave(self.recent4fitx_left)
            self.right_fitx = getave(self.recent4fitx_right)
            #print('average fit was used')
    
    def restoreback(self,img,oldimg,Minv,left_fitx,right_fitx):
            # Draw back to see if it make sense
            # Create an image to draw the lines on
            warp_zero = np.zeros_like(img[:,:,0]).astype(np.uint8)
            color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
            # Recast the x and y points into usable format for cv2.fillPoly()
            ploty = self.smoothy
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            #print(pts.shape,pts[0])
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
            result = cv2.addWeighted(oldimg, 1, newwarp, 0.3, 0)
            
            img2 = self.color_window(img[:,:,0])
            misc.imsave('./colored.jpg',img2)
            img2 = self.color_line(img2)
            misc.imsave('./colored_lined.jpg',img2)
            newwarp2 = cv2.warpPerspective(img2, Minv, (img.shape[1], img.shape[0]))
            result = cv2.addWeighted(result, 1, newwarp2, 0.3, 0)
            
            return result

    def find_window_centroids(self,warped):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-self.window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-self.window_width/2+int(warped.shape[1]/2)
        
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        # apply masking
        l_mask = self.window_mask(warped,window_centroids[0][0],0)
        r_mask = self.window_mask(warped,window_centroids[0][1],0)
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
        

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*self.window_height):int(warped.shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width/2
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+self.margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+self.margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))
            # Find left and right pts
            l_mask = self.window_mask(warped,window_centroids[level][0],level)
            r_mask = self.window_mask(warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
                
            
        #return window_centroids,window_centroids_y
        self.window_centroids = window_centroids
        
        templateL = np.array(l_points,np.uint8)
        templateR = np.array(r_points,np.uint8)
        
        self.leftpoints = templateL
        self.rightpoints = templateR
        
        Lpoints = templateL.nonzero()
        Rpoints = templateR.nonzero()
        
        # find the leftx and lefty
        # templateL = self.leftpoints
        leftx = np.array(Lpoints[1])
        lefty = np.array(Lpoints[0])
        # find the rightx and righty
        # templateR = self.rightpoints
        rightx = np.array(Rpoints[1])
        righty = np.array(Rpoints[0])
        # smooth fit the curve
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        ploty = self.smoothy
        self.left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        
    def window_slide_fit(self,img):
        '''
        Finds fits for current image and return an image 
        '''
        ploty = np.linspace(0,img.shape[0]-1,img.shape[0])
        self.smoothy = ploty
        # since image is binary, want to make it in the range of 0~255
        img = img*255
        # main pipline
        # find_window_centroids also fits
        self.find_window_centroids(img)
        # these two color function have little real impact on the final img
#        img = self.color_window(img)
#        self.find_fits()
#        img = self.color_line(img)
        img = np.dstack((img,img,img))
        return img
        
    def smallpipeline(self,img,oldimg,Minv):
        '''
        Check the fits done by window_slide_fit, perform either update or average fit
        Also restores back the original image
        '''
        self.sanitycheck()
        if self.detected == False:
            self.average_fit()
        else:
            self.update_recent_fit()
        
        img = self.restoreback(img,oldimg,Minv,self.left_fitx,self.right_fitx)
        
        # measure curvature and offset
        self.caloffset()
        # return img
        return img
    
    def lookahead(self,img):
        '''
        skip sliding window search if detected
        '''
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        
        templateL = np.zeros_like(img)        
        templateL[lefty,leftx] = 255
        self.leftpoints = templateL        

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        templateR = np.zeros_like(img)
        templateR[righty,rightx] = 255
        self.rightpoints = templateR

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = self.smoothy
        self.left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        self.right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        self.sanitycheck()
        img = np.dstack((img,img,img))
        return img
    
    def caloffset(self):
        '''
        This function calculates the offset between center in lanes and center
        in the image
        '''
        lane_center = (self.left_fitx[-1]+self.right_fitx[-1])/2
        img_center = 640
        self.offset = (lane_center-img_center)*self.xm_per_pix

    def window_mask(self, img_ref, center,level):
            output = np.zeros_like(img_ref)
            output2 = np.zeros_like(img_ref)
            output[int(img_ref.shape[0]-(level+1)*self.window_height):int(img_ref.shape[0]-level*self.window_height),max(0,int(center-self.window_width/2)):min(int(center+self.window_width/2),img_ref.shape[1])] = 1
            output2[(output > 0) & (img_ref > 0)] = 1
            #plt.imshow(output2,cmap='gray')
            return output2

    def color_window(self,img):

        # Draw the results
        zero_channel = np.zeros_like(img) # create a zero color channle 
        # define left and right points
        templateL = self.leftpoints
        templateR = self.rightpoints

        # draw left red, right blue
        template = np.array(cv2.merge((templateL,zero_channel,templateR)),np.uint8)
        warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(template, 0.8, warpage, 0.6, 0.0) # overlay the orignal road image with window results
        plt.imshow(output)                         
        return output
    
    def color_line(self,img):
        # draw the lines
        ploty = self.smoothy
        ptsleft = np.dstack((self.left_fitx,ploty))
        ptsright = np.dstack((self.right_fitx,ploty))
        cv2.polylines(img,np.int_(ptsleft),False,color=(150,0,255),thickness = 12)
        cv2.polylines(img,np.int_(ptsright),False,color=(255,0,0),thickness = 12)
        return img
   
    #this is not used
#    def find_fits(self):
#        # find the leftx and lefty
#        templateL = self.leftpoints
#        leftx = np.array(templateL[1])
#        lefty = np.array(templateL[0])
#        # find the rightx and righty
#        templateR = self.rightpoints
#        rightx = np.array(templateR[1])
#        righty = np.array(templateR[0])
#        # smooth fit the curve
#        self.left_fit = np.polyfit(lefty, leftx, 2)
#        self.right_fit = np.polyfit(righty, rightx, 2)
#        ploty = self.smoothy
#        self.left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
#        self.right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

    def sanitycheck(self):
        # decides self.detected
        # check for horiztonal distance
        hdist = []
        for i in range(len(self.left_fitx)):
            dist = abs(self.left_fitx[i]-self.right_fitx[i])
            hdist.append(dist)
        
        hdistave = np.array(hdist).mean()
        #self.hdistAve = hdistave
        # check for parallel (slope)
        tolerance = []
        for i in range(len(self.left_fitx)-1):
            slopeL = (self.smoothy[i]-self.smoothy[i+1])/(self.left_fitx[i]-self.left_fitx[i+1])
            slopeR = (self.smoothy[i]-self.smoothy[i+1])/(self.right_fitx[i]-self.right_fitx[i+1])
            if abs(slopeL/slopeR) < 1.4 and abs(slopeL/slopeR) > 0.6:
                tolerance.append(1)
            else:
                tolerance.append(0)
                
        slope_sim_level = tolerance.count(1)/len(tolerance)
        #print(slope_sim_level,)
        # check for curvature
        self.getcurvature()
        ratio = self.left_curverad/self.right_curverad
        if ratio < 0.5 or ratio > 1.7:
            curvestat = False
        else:
            curvestat = True
        #print(curvestat)
        # combine these conditions to make a judge
        if curvestat == True and slope_sim_level > 0.8 and (hdistave > 720 and hdistave < 880): # missing distance
            self.detected = True
            #print('good')
        else:
            self.detected = False
            #print('bad')
            
    def getcurvature(self):
        ploty = self.smoothy
        left_fit_cr = np.polyfit(ploty*self.ym_per_pix, self.left_fitx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*self.ym_per_pix, self.right_fitx*self.xm_per_pix, 2)
        # bottom of image
        # y_eval = np.max(ploty)
        y_eval = ploty[6*len(ploty)/7-1:len(ploty)]        
        # calculate new curvature
        self.left_curverad = (((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])).mean()
        self.right_curverad = (((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])).mean()
        self.curvature = max(self.left_curverad,self.right_curverad)*0.75+min(self.left_curverad,self.right_curverad)*0.25

    def line_main(self,img,oldimg,Minv):
        if self.detected == True:
            img = self.lookahead(img)
            if self.detected == False:
                img = self.window_slide_fit(img[:,:,0])
                img = self.smallpipeline(img,oldimg,Minv)
            else:
                img = self.restoreback(img,oldimg,Minv,self.left_fitx,self.right_fitx)
        else:
            img = self.window_slide_fit(img)
            img = self.smallpipeline(img,oldimg,Minv)
        
        return img

from scipy import misc            
from undistort_img import undistort_img as undistorter
from threshhold_img import threshhold_img as thresher
from warp_img import warp_img
import glob
        
if __name__ == "__main__":
    warper = warp_img()
    thresh = thresher()
    undist = undistorter()
    fitliner = line()
    count = 0

    images = glob.glob('../mytestpic/project_video.mp4_20170224*.jpg')
    #testimg = mpimg.imread('../test_images/test3.jpg')
    testimg = mpimg.imread(images[1])
    img = undist.cal_undistort(testimg)
    oldimg = np.copy(img)
    misc.imsave('./undistorted.jpg',img)
    img = thresh.sobel_color(img)
    misc.imsave('./thresholdedbinary.jpg',img*255)
    img = warper.getwarped(img)
    misc.imsave('./perspectivetransformed.jpg',img*255)
    img = fitliner.line_main(img,oldimg,warper.Minv)
    misc.imsave('./draw.jpg',img)
    #img2 = fitliner.restoreback(img2,oldimg,warper.Minv,fitliner.left_fitx,fitliner.right_fitx)
    
    print('average curv:',fitliner.curvature)
    print('offset is: %f meters' % fitliner.offset)

#    images = glob.glob('../mytestpic/project_video.mp4_20170224*.jpg')
#    picname = './restoredback'
#    for frame in images:
#        testimg = mpimg.imread(frame)
#        img = undist.cal_undistort(testimg)
#        oldimg = np.copy(img)
##        misc.imsave('./undistorted.jpg',img)
#        img = thresh.sobel_color(img)
##        misc.imsave('./thresholdedbinary.jpg',img*255)
#        img = warper.getwarped(img)
##        misc.imsave('./perspectivetransformed.jpg',img*255)
#        img = fitliner.line_main(img,oldimg,warper.Minv)
##        misc.imsave('./draw.jpg',img)
#        misc.imsave(picname+str(count)+'.jpg',img)
#        
#        #print(fitliner.left_curverad,fitliner.right_curverad)
#        print('Current count is:',count)
#        print('average curv:',fitliner.curvature)
#        print('offset is: %f meters' % fitliner.offset)
#        count += 1

    