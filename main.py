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

dataset = 'MNIST'
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
model_object = DCGAN(dataset, generator_arch, discriminator_arch, encoder_arch, learning_rate, batch_size)

model_object.train(x_train, epochs)


# def combine_images(generated_images):
#     num = generated_images.shape[0]
#     width = int(math.sqrt(num))
#     height = int(math.ceil(float(num) / width))
#     shape = generated_images.shape[1:3]
#     image = np.zeros((height * shape[0], width * shape[1]),
#                      dtype=generated_images.dtype)
#     for index, img in enumerate(generated_images):
#         i = int(index / width)
#         j = index % width
#         image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
#             img[:, :, 0]
#     return image


