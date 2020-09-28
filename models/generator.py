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
##

def residual_block(x):
    """
    Residual block
    """
    momentum = 0.8

    res = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu')(x)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Add()([res, x])

    return res

def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    """
    input_shape = (64, 64, 3)
    residual_blocks = 16
    momentum = 0.8

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    pre_res = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(pre_res)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    post_res = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    post_res = BatchNormalization(momentum=momentum)(post_res)

    # Take the sum of the output from the pre-residual blockand the post-residual block
    resnet_output = Add()([pre_res, post_res])

    # Add an upsampling block
    upsample_1 = UpSampling2D(size=2)(resnet_output)
    upsample_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(upsample_1)

    # Add another upsampling block
    up_sample2 = UpSampling2D(size=2)(upsample_1)
    up_sample2 = Conv2D(filters=256, kernel_size=3, padding='same' , activation= 'relu')(up_sample2)

    # Output convolution layer
    output = Conv2D(filters=3, kernel_size=9, padding='same', activation='tanh')(up_sample2)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model
