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

x_train, y_train, x_test, y_test = data_object.x_train, data_object.y_train, data_object.x_test, data_object.y_test


def run_test():
    model_object = DCGAN(dataset, generator_arch, discriminator_arch, encoder_arch, learning_rate, batch_size)

    if label is None:
        model_object.train(x_train, epochs)
    else:
        index = list(np.where(y_train[:, label] == 1)[0])
        x_positive = x_train[index]
        model_object.train(x_positive, epochs)


for label in range(10):
    run_test()



