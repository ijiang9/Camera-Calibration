#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:35:58 2020

@author: yinjiang
"""


import math
import numpy as np


def main():
    
    fw="./data/ncc-worldPt.txt"
    fi="./data/ncc-imagePt.txt"
    
    # f = "./data/points.txt"
    # data = np.loadtxt(f)
    # objp = data[:,:3]
    # imagepoints = data[:,3:]
    
    imagepoints=np.loadtxt(fi)
    objp=np.loadtxt(fw)
    new_col=np.ones((len(objp),1),np.float32)
    objp = np.append(objp, new_col, 1)
    
    A=matrixA(objp, imagepoints)
    M=calculatePara(A)
    errorcal(objp,imagepoints,M)
    #cvcalibrate(filename)
def errorcal(objp, imagep, M):
    # error calculating
    m1=M[0]
    m2=M[1]
    m3=M[2]
    error=0
    for i in range(objp.shape[0]):
        const=m3.T.dot(objp[i])
        xi=imagep[i][0]
        yi=imagep[i][1]
        xip=m1.T.dot(objp[i])/const
        yip=m2.T.dot(objp[i])/const
        error=error+(xi-xip)**2+(yi-yip)**2
    error=error/objp.shape[0]
    print("error:")
    print(error)
        
        
        
def matrixA(objp, imagep): 
    #calculate matrix A
    A = np.zeros((len(objp)*2, 12))
    zero = np.zeros(4)
    for i in range(0,len(objp)):
        pi=objp[i]
        xipi=-imagep[i][0]*pi
        yipi=-imagep[i][1]*pi
        A[2*i]=np.concatenate([pi, zero, xipi])
        A[2*i+1]=np.concatenate([zero, pi, yipi])   
    return A


def calculatePara(A):
    #calculate parameters
    np.set_printoptions(formatter={'float': "{0:.6f}".format})
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    M=vh[-1].reshape(3, 4)
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    b = []
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1))
    
    rho=1/np.linalg.norm(a3)
    u0=rho**2*np.dot(a1,a3)
    v0=rho**2*np.dot(a2,a3)
    av=math.sqrt(rho**2*np.dot(a2,a2)-v0**2)
    a1xa3 = np.cross(a1, a3)
    a2xa3 = np.cross(a2, a3)
    #s=0
    s = (rho ** 4) / av * a1xa3.dot(a2xa3.T)
    au=math.sqrt(rho**2*a1.dot(a1.T)-s**2-u0**2)
    Ks = np.array([[au, s, u0],[0, av, v0],[0, 0, 1]])
    sign = np.sign(b[2])
    print("(u0,v0)=")
    print("(%f,    %f)" %(u0,v0))
    
    print("(alphaU,alphaV)")
    print("(%f,    %f)" %(au,av))
    print("s=")
    print(s)
    
    Ts=sign*rho*np.linalg.inv(Ks).dot(b).T
    print("T* =")
    print(Ts)
    r3=sign*rho*a3
    r1=rho**2/av*a2xa3
    r2=np.cross(r3,r1)
    Rs=np.array([r1,r2,r3])
    print("R* =")
    print(Rs)
    return M


if __name__ == '__main__':
    main()