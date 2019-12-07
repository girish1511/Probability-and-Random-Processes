#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:11:07 2018

@author: girish
"""

import numpy as np
import cv2
def disp_img(im,r,c,d):
    img=np.reshape(im,(r,c,d)).astype(np.uint8)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',cv2.resize(img,(560,460)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return
