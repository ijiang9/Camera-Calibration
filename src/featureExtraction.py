#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:03:40 2020

@author: yinjiang
"""


import cv2
import numpy as np


def main():
    print(__doc__)
    image = getImage()
    extractFeats(image)


def getImage():
    image = cv2.imread('./data/target.jpg')
    return image;

def extractFeats(image):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1,6*9,2), np.float32)
    objp[0,:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)*10
    # add 3-d coordinate
    col = []
    d=1
    for i in range(6):
        for j in range(9):
            col.append(d)
        d += 1
    col = np.array(col, np.float32)
    col = col.reshape((54,1))*10
    dd = np.concatenate([objp[0],col],axis=1)
    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpoints = [] # 2d points in image plane.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret:
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (9,6), corners2, ret)
        cv2.imshow("image", image)
        objp = np.zeros((1,54,3),np.float32)
        objp[0,:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
        objpoints.append(objp)
        rett,mtx = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
        # np.savetxt('./data/ip.txt',corners2.reshape(-1,2))
        # np.savetxt('./data/op.txt', objp[0])
        # np.savetxt('./data/op.txt', dd)
        data = np.concatenate([dd,corners2.reshape(-1,2)],axis = 1)
        np.savetxt('./data/points.txt',data)

        cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()