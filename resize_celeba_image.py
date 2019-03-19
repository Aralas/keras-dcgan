# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:cut_celeba_image.py
@time:2019-03-1816:03
"""

import cv2
import os

list1 = os.listdir("./dataset/img_align_celeba/")
n = len(list1)
for i in range(n):
    print(i, '/', n)
    imgName = os.path.basename(list1[i])
    if (os.path.splitext(imgName)[1] != ".jpg"): continue

    img = cv2.imread('dataset/img_align_celeba/' + imgName)

    img_nearest = cv2.resize(img, (160, 192), cv2.INTER_NEAREST)

    cv2.imwrite('dataset/resized_celeba/' + imgName, img_nearest)
