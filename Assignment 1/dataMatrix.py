#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:43:34 2018

@author: girish
"""
import cv2
import numpy as np
image=cv2.imread('test.jpg')
def dataMatrix(im):
    sz=np.shape(im);
    r=sz[0]
    c=sz[1]
    if len(sz)==3:
        d=sz[2]
    else:
        d=1
        temp=np.zeros((r,c,d))
        temp[:,:,0]=im
        im=temp
    data=np.zeros((r*c*d,1))
    for k in range(d):
        for j in range(c):
            for i in range(r):
                data[i+(r*j)+(r*c*k),0]=im[i,j,k]
            
    return data

gr=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
da=dataMatrix(gr)