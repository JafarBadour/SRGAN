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
from .generator import build_generator
from .discriminator import build_discriminator




def build_vgg():
    """
    Build VGG network to extract image features
    """
    input_shape = (256, 256, 3)

    vgg = keras.applications.VGG19(include_top = False ,  input_shape = input_shape , weights="imagenet")
    features = vgg.get_layer(index = 9).output

    model = keras.Model(inputs=[vgg.inputs], outputs=[features])
    return model


def define_adv_model(board_logs_path, logdir, generated_high_resolution_images):
    logdir = os.path.join(board_logs_path , datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = keras.callbacks.TensorBoard(logdir , histogram_freq=1 , profile_batch=0)
    writer = tf.summary.create_file_writer(logdir)

    mode = 'predict'

    epochs = 30001
    batch_size = 1

    common_optimizer = Adam(0.0002, 0.5)
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    vgg = build_vgg()
    vgg.trainable = False
    vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # Build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # Build the generator network
    generator = build_generator()

    """
    Build and compile the adversarial model
    """

    # Input layers for high-resolution and low-resolution images
    input_high_resolution = Input(shape=high_resolution_shape)
    input_low_resolution = Input(shape=low_resolution_shape)

    # Generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator(input_low_resolution)

    # Extract feature maps of the generated images
    features = vgg(generated_high_resolution_images)

    # Get the probability of generated high-resolution images
    probs = discriminator(generated_high_resolution_images)

    # Create and compile an adversarial model
    adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)

    return adversarial_model
