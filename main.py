# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:main.py.py
@time:2019-03-1515:11

This code implements DCGAN + autoencoder.

Parameters:
    code_dim - length of random vector (input of generator)
    generator_arch - architecture of generator ([number of filters] in convolutional layers)
    discriminator_arch - architecture of discriminator ([[filters, strides] and hidden units] in the last fc layer)
    encoder_arch -


"""
from dcgan import DCGAN
import DataProcessing as DP
import numpy as np
import tensorflow as tf
from keras import backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

dataset = 'MNIST'
label = 0
generator_arch = [128, 64]
discriminator_arch = [[32, 2], [64, 2], [128, 2], [256, 1]]
encoder_arch = []
learning_rate = 0.0002
batch_size = 128
epochs = 20

if dataset == 'MNIST':
    data_object = DP.MNIST()
elif dataset == 'CIFAR10':
    data_object = DP.CIFAR10()

if label is None:
    class_name = 'whole'
else:
    class_name = data_object.class_list[label]
    
x, y = data_object.x, data_object.y


def run_test():
    model_object = DCGAN(dataset, label, data_object.input_size, class_name, generator_arch, discriminator_arch, encoder_arch, learning_rate, batch_size)

    if label is None:
        model_object.train(x, epochs)
    else:
        index = list(np.where(y[:, label] == 1)[0])
        x_positive = x[index]
        model_object.train(x_positive, epochs)


for label in range(10):
    run_test()



