#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:10:52 2018

@author: girish
"""
"""Face Recognition"""
#%%Training

import numpy as np
import cv2

def disp_img(im,r,c,d):
    img=np.reshape(im,(r,c,d)).astype(np.uint8)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',cv2.resize(img,(560,460)))
    cv2.imwrite('eigenface.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return

"""Reading all training images"""
images=[]
for i in range(40):
    for j in range (8):
        images.append(cv2.imread('att_faces/s'+str(i+1)+'/'+str(j+1)+'.pgm'))
#images=np.array([cv2.imread(file) for file in glob.glob('/Users/girish/Desktop/Training/*.pgm')])
sz=np.shape(images)
ns=40       #No of subjects
ni=sz[0]    #No of Images
r=sz[1]
c=sz[2]
d=sz[3]
"""Converting image matrix into vector and creating the data matrix"""
data=np.zeros((r*c*d,ni)).astype(np.uint8)
for j in range(ni):
    data[:,j]=np.matrix.flatten(images[j]).T
    
"""Calculating the mean image"""
exp=np.zeros((r*c*d,1))
exp=np.mean(data,axis=1)

"""Calculating the difference image(Actual-Mean)"""
A=np.zeros((r*c*d,ni))
for k in range(ni):
    A[:,k]=data[:,k]-exp

"""Covariance Matrix, EigenVectors and Eigenvalues"""
cov=np.matmul(A.T,A)
cov=np.divide(cov,ni)
eigval,eigvec=np.linalg.eig(cov)
ef=np.matmul(A,eigvec)
"""Sorting the eigenvectors based on eigenvalues"""
indsort=eigval.argsort()
ev=eigval[indsort[::-1]]
ef=ef[:,indsort[::-1]]

k=ni-1

"""Normalization of Eigenfaces"""
for i in range(int(k)):
    ef[:,i]=np.divide(ef[:,i],np.linalg.norm(ef[:,i]))
efp=ef+abs(ef.min())
efd=np.divide(efp,efp.max())*255

"""Finding the weights of each image"""
w=np.zeros((ni,r*c*d))
w=np.matmul(ef.T,A)

"""Uncomment the follwoing to display image.
Inplace of imag input the image to be displayed in vector or matrix form"""
#disp_img(imag,r,c,d)  
#%%Testing
"""Reading the test images"""
test=[]
ns=40   #No of subjects
for i in range(ns):
    for j in range(8,10):
        test.append(cv2.imread('att_faces/s'+str(i+1)+'/'+str(j+1)+'.pgm'))

nt=int((len(test)/ns))  #No of test images per subject

"""Creating the data matrix for test images"""
dt=np.zeros((r*c*d,ns*nt))
for i in range(ns*nt):
    dt[:,i]=np.matrix.flatten(test[i]).T-exp
"""Calculating the weights of each test image"""
wt=np.matmul(ef.T,dt)

"""Finding the error as minimum of norm of difference between weights of training and testing images"""
dif=np.zeros((int(k),ns*nt))
for i in range(int(k)):
    for j in range(ns*nt):
        dif[i,j]=np.linalg.norm(w[:,i]-wt[:,j])

"""Calculating and displaying the accuracy of prediction"""
pred=np.zeros((ns*nt,1))
for i in range(ns*nt):
    pred[i,0]=(np.argmin(dif[:,i])/8+1).astype(np.uint8)
pred=pred.astype(np.uint8)
pred=pred.reshape(ns,nt)
count=0
for i in range(ns):
    for j in range(nt):
        if i+1==pred[i,j]:
            count+=1
acc=count*100/(ns*nt)
print('Accuracy =',acc,'%')











