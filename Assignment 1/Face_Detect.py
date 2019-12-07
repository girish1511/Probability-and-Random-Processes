#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:14:38 2018

@author: girish
"""
"""Face Detection"""
#%%Training

import numpy as np
import cv2
from matplotlib import pyplot as plt

def disp_img(im,r,c,d):
    img=np.reshape(im,(r,c,d)).astype(np.uint8)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',cv2.resize(img,(560,460)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return

"""Reading all training images"""
images=[]
for i in range(35):
    for j in range (10):
        images.append(cv2.imread('att_faces/s'+str(i+1)+'/'+str(j+1)+'.pgm'))
#images=np.array([cv2.imread(file) for file in glob.glob('/Users/girish/Desktop/Training/*.pgm')])
sz=np.shape(images)
nstr=20       #No of subjects
ni=sz[0]      #No of Images
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

k=ni

"""Normalization of Eigenfaces"""
for i in range(int(k)):
    ef[:,i]=np.divide(ef[:,i],np.linalg.norm(ef[:,i]))
efp=ef+abs(ef.min())
efd=np.divide(efp,efp.max())*255

"""Finding the weights of each image"""
w=np.zeros((ni,r*c*d))
w=np.matmul(ef.T,A)

"""Uncomment the follwoing to display image.
Inplace of img input the image to be displayed in vector or matrix form"""
disp_img(exp,r,c,d)  
#%%Testing
"""Reading the test images"""
test=[]
ns=5   #No of subjects
nt=10  #No of test images per subject
for i in range(ns):
    for j in range(nt):
        test.append(cv2.imread('att_faces/s'+str(i+36)+'/'+str(j+1)+'.pgm'))

  #No of test images per subject

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
er=dif.min(axis=0)
       

t=np.zeros((7501,1))
for i in range(0,7501):
    t[i]=i
nthr=len(t)
acc=np.zeros((nthr,1))
"""Calculating accuracies of detection for a range of threshold values"""
for i in range(nthr):
    pred=np.ones((ns*nt,1),dtype=bool)
    pred=er<t[i]
    count=0
    for j in range(len(pred)):
        if pred[j]:
            count+=1
            acc[i,0]=count*100/(ns*nt)
"""Plotting Thershold vs Accuracy"""
plt.title("Accuracy vs Threshold") 
plt.xlabel("Threshold") 
plt.ylabel("Accuracy(%)") 
plt.plot(t,acc) 
plt.plot(t,100-acc)
plt.show()

"""Considering a threshold value of T=6000 accuracy is printed"""
T=6000
pred=np.ones((ns*nt,1),dtype=bool)
pred=er<T
count=0
for j in range(len(pred)):
    if pred[j]:
        count+=1
    accT=count*100/(ns*nt)
pred=pred.reshape(ns,nt)
print('Accuracy for threshold of 6000 =',accT,'%')