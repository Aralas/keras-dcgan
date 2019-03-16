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
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

    def load_data(self):
        mnist = tf.contrib.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        # transform labels to one-hot vectors
        y_train = tf.contrib.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.contrib.keras.utils.to_categorical(y_test, self.num_classes)
        return x_train, y_train, x_test, y_test


class CIFAR10(object):

    def __init__(self, seed, noise_level, augmentation):
        LoadData.__init__(self, seed, noise_level, augmentation)
        self.num_classes = 10
        self.img_rows, self.img_cols = 32, 32
        self.input_size = (32, 32, 3)
        self.x_train, self.y_train, self.y_train_orig, self.x_test, self.y_test, self.clean_index = self.data_preprocess()

    def load_data(self):
        # load data
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 3)

        # transform labels to one-hot vectors
        y_train = tf.contrib.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.contrib.keras.utils.to_categorical(y_test, self.num_classes)
        return x_train, y_train, x_test, y_test


class Fruit360(object):

    def __init__(self, seed, noise_level, augmentation):
        LoadData.__init__(self, seed, noise_level, augmentation)
        self.num_classes = 95
        self.img_rows, self.img_cols = 64, 64
        self.input_size = (64, 64, 3)
        self.x_train, self.y_train, self.y_train_orig, self.x_test, self.y_test, self.clean_index = self.data_preprocess()

    def load_images(self, path):
        img_data = []
        labels = []
        idx_to_label = []
        i = -1
        for fruit in os.listdir(path):
            if not fruit.startswith('.'):
                fruit_path = os.path.join(path, fruit)
                labels.append(fruit)
                i = i + 1
                for img in os.listdir(fruit_path):
                    img_path = os.path.join(fruit_path, img)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (64, 64))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    img_data.append(image)
                    idx_to_label.append(i)
        return np.array(img_data), np.array(idx_to_label), labels

    def load_data(self):
        # load data
        trn_data_path = 'fruits-360/Training'
        val_data_path = 'fruits-360/Test'
        x_train, y_train, label_data = self.load_images(trn_data_path)
        x_test, y_test, label_data_garbage = self.load_images(val_data_path)
        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test
