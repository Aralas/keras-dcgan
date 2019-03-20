from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import math
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


class DCGAN():

    def __init__(self, dataset, label, image_shape, class_name, generator_arch, discriminator_arch, encoder_arch, learning_rate, batch_size):
        self.dataset = dataset
        self.label = label
        self.generator_arch = generator_arch
        self.discriminator_arch = discriminator_arch
        self.encoder_arch = encoder_arch

        self.image_shape = image_shape
        self.class_name = class_name
        self.image_row, self.image_column, self.image_channel = self.image_shape
        self.code_dim = 100
        self.kernel_size = (5, 5)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = Adam(lr=self.learning_rate)

        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.encoder = self.encoder_model()
        

    def data_generator(self, x, batch_size):
        idx = np.arange(len(x))
        while True:
            np.random.shuffle(idx)
            batches = [idx[range(batch_size * i, min(len(x), batch_size * (i + 1)))] for i in
                       range(math.ceil(len(x) / batch_size))]
            for index in batches:
                yield x[index]

    def generator_model(self):
        architecture = self.generator_arch
        fractor = np.power(2, len(architecture))
        dim1 = self.image_row // fractor
        dim2 = self.image_column // fractor
        
        if max(self.image_row % fractor, self.image_column % fractor) != 0:
            print('invalid architecture')
            return
        
        if min(dim1, dim2) < 4:
            print('too many layers')
            return
        
        model = Sequential()
        for layer_index in range(len(architecture)):
            filter_num = architecture[layer_index]
            if layer_index == 0:
                model.add(Dense(input_dim=self.code_dim, output_dim=dim1 * dim2 * filter_num, activation='relu'))
                model.add(BatchNormalization(momentum=0.8))
                model.add(Reshape((dim1, dim1, filter_num)))
                model.add(UpSampling2D(size=(2, 2)))
            else:
                model.add(Conv2D(filter_num, kernel_size=self.kernel_size, kernel_initializer='glorot_normal',
                                 padding='same', activation='relu'))
                model.add(BatchNormalization(momentum=0.8))
                model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(self.image_channel, self.kernel_size, kernel_initializer='glorot_normal', padding='same',
                         activation='tanh'))
        return model

    def discriminator_model(self):
        model = Sequential()
        architecure = self.discriminator_arch
        for layer_index in range(len(architecure)):
            layer = architecure[layer_index]
            if len(layer) != 2:
                print('invalid architecuture')
                return
            if layer_index == 0:
                model.add(Conv2D(layer[0], kernel_size=self.kernel_size, input_shape=self.image_shape,
                                 strides=layer[1], kernel_initializer='glorot_normal', padding='same'))
            else:
                model.add(Conv2D(layer[0], kernel_size=self.kernel_size, strides=layer[1],
                                 kernel_initializer='glorot_normal', padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            # model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def encoder_model(self):
        model = Sequential()
        return model

    def generator_containing_discriminator(self):
        model = Sequential()
        model.add(self.generator)
        self.discriminator.trainable = False
        model.add(self.discriminator)
        return model

    def encoder_containing_generator(self):
        model = Sequential()
        model.add(self.encoder)
        self.generator.trainable = False
        model.add(self.generator)
        return model

    def train(self, x_train, epochs):
        # d = self.discriminator
        # g = self.generator
        g_plus_d = self.generator_containing_discriminator()

        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        g_plus_d.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        batch_num = math.ceil(len(x_train) / self.batch_size)
        data_generator = self.data_generator(x_train, self.batch_size)

        for epoch in range(epochs):
            print("Epoch is", epoch)
            for batch in range(batch_num):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                noise = np.random.uniform(-1, 1, size=(self.batch_size, self.code_dim))
                x_from_generator = self.generator.predict(noise)
                x_true = next(data_generator)
                x = np.concatenate((x_true, x_from_generator))
                y = [1] * x_true.shape[0] + [0] * self.batch_size
                d_loss = self.discriminator.train_on_batch(x, y)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = np.random.uniform(-1, 1, size=(self.batch_size, self.code_dim))
                self.discriminator.trainable = False
                g_loss = g_plus_d.train_on_batch(noise, [1] * self.batch_size)
                self.discriminator.trainable = True
                print("batch: ", batch, "/", batch_num, ", d_loss: ", d_loss[0], ", g_loss: ", g_loss[0])

            # ---------------------
            #  Save Images
            # ---------------------
            noise = np.random.uniform(-1, 1, size=(10 * 10, self.code_dim))
            images = self.generator.predict(noise)
            images = (images + 1) * 127.5

            noise_nice = np.random.uniform(-1, 1, size=(2000, self.code_dim))
            images_nice = self.generator.predict(noise_nice)
            images_nice = (images_nice + 1) * 127.5
            d_predict = self.discriminator.predict(images_nice)
            d_pred_sort_index = np.argsort(-d_predict, axis=0).reshape(-1, )

            fig, axs = plt.subplots(10, 10)
            cnt = 0
            for i in range(10):
                for j in range(10):
                    if self.dataset == 'MNIST':
                        axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
                    else:
                        r = Image.fromarray(images[cnt, :, :, 0]).convert('L')
                        g = Image.fromarray(images[cnt, :, :, 1]).convert('L')
                        b = Image.fromarray(images[cnt, :, :, 2]).convert('L')
                        img = Image.merge('RGB', (r, g, b))
                        axs[i, j].imshow(img)
                    axs[i, j].axis('off')
                    cnt += 1
            dirs = 'images/random/' + self.dataset + '_' + self.class_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            fig.savefig(dirs + '/epoch_' + str(epoch) + '.png')
            plt.close()

            fig, axs = plt.subplots(10, 10)
            cnt = 0
            for i in range(10):
                for j in range(10):
                    index_nice = d_pred_sort_index[cnt]
                    if self.dataset == 'MNIST':
                        axs[i, j].imshow(images_nice[index_nice, :, :, 0], cmap='gray')
                    else:
                        r = Image.fromarray(images_nice[index_nice, :, :, 0]).convert('L')
                        g = Image.fromarray(images_nice[index_nice, :, :, 1]).convert('L')
                        b = Image.fromarray(images_nice[index_nice, :, :, 2]).convert('L')
                        img = Image.merge('RGB', (r, g, b))
                        axs[i, j].imshow(img)
                    axs[i, j].axis('off')
                    cnt += 1
            dirs = 'images/nice/' + self.dataset + '_' + self.class_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            fig.savefig(dirs + '/epoch_' + str(epoch) + '.png')
            plt.close()

            # ---------------------
            #  Save Models
            # ---------------------
            if epoch % 50 == 49:
                self.generator.save_weights('model/' + self.dataset + '_' + self.class_name + '_generator_epoch_' + str(epoch) + '.hs')
                self.discriminator.save_weights('model/' + self.dataset + '_' + self.class_name + '_discriminator_epoch_' + str(epoch) + '.hs')
