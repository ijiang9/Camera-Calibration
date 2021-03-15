#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 03:37:59 2020

@author: yinjiang
"""


import numpy as np

import math

def main():
    fw="./data/ncc-worldPt.txt"
    fi="./data/ncc-noise-1-imagePt.txt"
    ip=np.loadtxt(fi,skiprows=1)
    op=np.loadtxt(fw,skiprows=1)
    new_col=np.ones((len(op),1),np.float32)
    op = np.append(op, new_col, 1)
    n, d, kmax = config()
    M, inlier=ransac(op,ip,n,d,kmax)
    # print(inlinerNum, bestM)
    computeParams(M)
    fc="./data/ncc-imagePt.txt"
    ipc=np.loadtxt(fc,skiprows=1)
    err = 0
    for i, j in zip(ip,ipc):
        err = err + (i[0]-j[0])**2+(i[1]-j[1])**2
    print("error compare to ncc-image: %f" %err)
    
def computeParams(M):
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
    

def ransac(objp, imagep, n, d, kmax):
    k=kmax
    w=0.5
    iteration=0
    inlier=list(range(0,len(objp)))
    A=matrixA(objp,imagep)
    bestfit=matrixM(A)
    dist=distance(objp,imagep,bestfit)
    besterr=error(dist)
    while iteration<k and iteration<kmax:
        index=np.random.choice(len(objp),n,replace = True)
        ranobjp, ranimgp=objp[index],imagep[index]
        A=matrixA(ranobjp,ranimgp)
        maybemodel=matrixM(A)
        alsoinlier=[]
        dist=distance(ranobjp,ranimgp,maybemodel)
        t=1.5*np.median(dist)
        for i in list(set(range(0,len(objp)))-set(index)):
            if distancep(objp[i],imagep[i],maybemodel)<t:
                alsoinlier.append(i)
        if len(alsoinlier)>d:
            # bettermodel=maybemodel
            index=list(index)+alsoinlier
            op_t, ip_t = objp[index],imagep[index]
            A_t = matrixA(op_t,ip_t)
            bettermodel = matrixM(A_t)
            dist=distance(op_t,ip_t,bettermodel)
            thiserr=error(dist)
            if thiserr<besterr:
                bestfit=bettermodel
                besterr=thiserr
                inlier=index
        w=float(len(inlier))/len(objp)
        if w>0 and w<1:
            k=math.log(1-0.99)/math.log(1-w**n)
        iteration=iteration+1
    return bestfit,inlier
def error(d):
    return d.dot(d.T)/d.shape[0]
def distance(objp, imagep, M):
   
    d=[]
    for i in range(objp.shape[0]):
        distance=distancep(objp[i],imagep[i],M)
        d.append(distance)
    return np.array(d)
def distancep(op,ip,M):
    #return distance of one point pair
    m1=M[0]
    m2=M[1]
    m3=M[2]
    const=m3.T.dot(op)
    xi=ip[0]
    yi=ip[1]
    xip=m1.T.dot(op)/const
    yip=m2.T.dot(op)/const
    distance=math.sqrt((xi-xip)**2+(yi-yip)**2)
    return distance

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

def matrixM(A):
    M = []
    u, s, v = np.linalg.svd(A, full_matrices = True)
    M = v[-1].reshape(3, 4)
    return M


def config():
    configname = 'RANSAC.config'
    with open(configname, 'r') as conf:
        conf.readline()
        n = int(conf.readline().split()[0])
        d = int(conf.readline().split()[0])
        kmax = int(conf.readline().split()[0])
    return n, d, kmax


if __name__ == '__main__':
    main()
