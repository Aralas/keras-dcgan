# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:cut_celeba_image.py
@time:2019-03-1816:03
"""

from PIL import Image
import face_recognition
import os

list1 = os.listdir("./dataset/img_align_celeba/")
list2 = os.listdir('./dataset/new_celeba/')
list = set(list1) - set(list2)
n = len(list)
for i in range(n):
    print(i, '/', n)
    imgName = os.path.basename(list[i])
    if (os.path.splitext(imgName)[1] != ".jpg"): continue

    image = face_recognition.load_image_file('./dataset/img_align_celeba/' + imgName)

    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        width = right - left
        height = bottom - top
        if (width > height):
            right -= (width - height)
        elif (height > width):
            bottom -= (height - width)
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save('./dataset/new_celeba/' + imgName)

