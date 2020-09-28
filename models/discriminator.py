import tensorflow as tf
from tensorflow import keras
from PIL import Image
from PIL.ImageColor import getrgb
import glob
import os
import matplotlib.pyplot as plt##
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

def build_discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Add the first convolution block
    first_conv = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    first_conv = LeakyReLU(alpha=leakyrelu_alpha)(first_conv)

    filters , strides = [64,128,128,256,256,512,512] , [2,1,2,1,2,1,2]
    cur = first_conv

    for f , s in zip(filters , strides):
        cur = Conv2D(filters=f, kernel_size=3, strides=s, padding='same')(cur)
        cur = LeakyReLU(alpha=leakyrelu_alpha)(cur)
        cur = BatchNormalization(momentum=momentum)(cur)

    flat = keras.layers.Flatten()(cur)
    dense = Dense(units=1024)(flat)
    dense = LeakyReLU(alpha=0.2)(dense)

    # Last dense layer - for classification
    output = Dense(units=1, activation='sigmoid')(dense)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')

    return model