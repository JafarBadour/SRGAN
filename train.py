from models.methods import just_train

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
from models.gan import build_gan,build_vgg
from utils import input_pipeline, save_images



training_images_path = "./training/"
testing_images_path = "./testing/"
res_im_path = "./results/"
board_logs_path = "./logs/"
models_path = "./models/"

os.makedirs(res_im_path , exist_ok= True)
os.makedirs(board_logs_path , exist_ok = True)
os.makedirs(models_path, exist_ok = True)


def just_train():
    logdir = os.path.join(board_logs_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)
    dataloader = iter(input_pipeline(training_images_path, 1, high_resolution_shape[:2], low_resolution_shape[:2]))
    # Build and compile VGG19 network to extract features
    epochs = 30001
    batch_size = 1
    vgg = build_vgg()
    gan , generator , discriminator = build_gan()
    writer = tf.summary.create_file_writer(logdir)

    for epoch in range(epochs):

        """
        Train the discriminator network
        """

        # Sample a batch of images
        # high_resolution_images, low_resolution_images = next(highres_dataloader) , next(lowres_dataloader)
        high_resolution_images, low_resolution_images = next(dataloader)

        # Generate high-resolution images from low-resolution images
        generated_high_resolution_images = generator.predict(low_resolution_images)

        # Generate batch of real and fake labels

        # real_labels = np.ones((batch_size, 16, 16, 1))
        # fake_labels = np.zeros((batch_size, 16, 16, 1))

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator network on real and fake images
        d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

        # Calculate total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # print("d_loss:", d_loss)

        """
        Train the generator network
        """

        # Sample a batch of images
        high_resolution_images, low_resolution_images = next(dataloader)

        # Extract feature maps for real high-resolution images
        image_features = vgg.predict(high_resolution_images)

        # Train the generator network
        g_loss = gan.train_on_batch([low_resolution_images, high_resolution_images],
                                                  [real_labels, image_features])

        # print("g_loss:", g_loss)

        print("Epoch {} : g_loss: {} , d_loss: {}".format(epoch, g_loss[0], d_loss[0]))


        # Sample and save images after every 100 epochs
        with writer.as_default():
            tf.summary.scalar('g_loss', g_loss[0], step=epoch)
            tf.summary.scalar('d_loss', d_loss[0], step=epoch)
        writer.flush()


        print("Epoch {}:  Gloss : {} , Dloss {}".format(epoch+1 , g_loss , d_loss))
        if (epoch) % 100 == 0:

            high_resolution_images, low_resolution_images = next(dataloader)
            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(res_im_path + "img_{}_{}".format(epoch, index),
                            low_resolution_images[index], generated_images[index], high_resolution_images[index]
                            )

        if (epoch) % 1000 == 0:
            # Save models
            generator.save_weights(models_path + "generator_{}.h5".format(epoch))
            discriminator.save_weights(models_path + "discriminator_{}.h5".format(epoch))



if __name__ == '__main__':
    just_train()
    ##