import tensorflow as tf
from tensorflow import keras
from PIL import Image
from PIL.ImageColor import getrgb
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, Flatten
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import shutil
import datetime


def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res

def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model


def input_pipeline(data_path, batch_size, highres_shape, lowres_shape):
    all_images = glob.glob(data_path + "*")

    cntall = len(all_images)

    def gen():

        while True:

            all = []
            all_highres = []
            all_lowres = []

            idxes = np.random.choice(cntall, batch_size, replace=False)

            for idx in idxes:

                fname = all_images[idx]

                orig = Image.open(fname)

                # orig = orig.astype(np.float32)

                high_img = orig.resize(highres_shape, resample=Image.BICUBIC)
                low_img = orig.resize(lowres_shape, resample=Image.BICUBIC)

                if np.random.random() < 0.5:
                    high_img = np.fliplr(high_img)
                    low_img = np.fliplr(low_img)

                all_highres.append(np.asarray(high_img, dtype=np.float32))
                all_lowres.append(np.asarray(low_img, dtype=np.float32))

                high_res_ret = np.array(all_highres) / 127.5 - 1
                low_res_ret = np.array(all_lowres) / 127.5 - 1

            yield (high_res_ret, low_res_ret)

    return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)).prefetch(5)


def save_images(data_path, lowres, highres, orig):
    lowres = np.squeeze((lowres.numpy() + 1) / 2.0)
    highres = np.squeeze((highres + 1) / 2.0)
    orig = np.squeeze((orig.numpy() + 1) / 2.0)

    # blank = np.ones((orig.shape[0] , 30 , 3) , dtype = np.float32)

    # print(img1.shape , blank.shape , img2.shape)

    # tot_img = np.hstack((lowres , blank , highres , blank , orig))
    # tot_img = np.clip(tot_img , 0.0001 , 0.9999)

    fig = plt.figure(figsize=(12, 4))

    # low_resolution_image = low_resolution_image * 0.5 + 0.5
    # generated_image = generated_image * 0.5 + 0.5

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(lowres)
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(orig)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(highres)
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(data_path)

    # plt.imsave(data_path , tot_img)