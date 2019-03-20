# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:DataProcessing.py.py
@time:2019-03-1516:33
"""

import tensorflow as tf
import os
import numpy as np
from keras.utils import np_utils
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


class MNIST(object):

    def __init__(self):
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28
        self.input_size = (28, 28, 1)
        self.x, self.y = self.load_data()
        self.class_list = ['digit0', 'digit1', 'digit2', 'digit3', 'digit4', 'digit5', 'digit6', 'digit7', 'digit8',
                          'digit9']

    def load_data(self):
        mnist = tf.contrib.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x = x/127.5 - 1
        x = x.reshape(x.shape[0], self.img_rows, self.img_cols, 1)
        
        # transform labels to one-hot vectors
        y = tf.contrib.keras.utils.to_categorical(y, self.num_classes)
        return x, y


class CIFAR10(object):

    def __init__(self):
        self.num_classes = 10
        self.img_rows, self.img_cols = 32, 32
        self.input_size = (32, 32, 3)
        self.x, self.y = self.load_data()
        self.class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load_data(self):
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x = x/127.5 - 1
        x = x.reshape(x.shape[0], self.img_rows, self.img_cols, 3)
        
        # transform labels to one-hot vectors
        y = tf.contrib.keras.utils.to_categorical(y, self.num_classes)
        return x, y


class celeba(object):

    def __init__(self):
        self.num_classes = None
        self.img_rows, self.img_cols = 192, 160
        self.input_size = (192, 160, 3)
        self.x, self.y = self.load_data()
        self.class_list = None

    def load_data(self):
        list1 = os.listdir("./dataset/resized_celeba/")
        n = len(list1)
        x = np.zeros((n, self.img_rows, self.img_cols, 3))
        y = None
        for i in range(n):
            imgName = os.path.basename(list1[i])
            if (os.path.splitext(imgName)[1] != ".jpg"): continue
            img = np.array(cv2.imread('dataset/img_align_celeba/' + imgName))
            x[i, :, :, :] = img
        return x, y

