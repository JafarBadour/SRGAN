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


def just_train(training_images_path,high_resolution_shape,low_resolution_shape, epochs, generator, discriminator, vgg, adversarial_model, writer, batch_size, res_im_path, models_path):
    dataloader = iter(input_pipeline(training_images_path, 1, high_resolution_shape[:2], low_resolution_shape[:2]))
    # Build and compile VGG19 network to extract features

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
        real_labels = np.ones((batch_size, 16, 16, 1))
        fake_labels = np.zeros((batch_size, 16, 16, 1))

        # Train the discriminator network on real and fake images
        d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

        # Calculate total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # print("d_loss:", d_loss)

        """
        Train the generator network
        """
        ##
        # Sample a batch of images

        # high_resolution_images, low_resolution_images = sample_images(batch_size)
        # high_resolution_images, low_resolution_images = next(highres_dataloader) , next(lowres_dataloader)
        high_resolution_images, low_resolution_images = next(dataloader)

        # Extract feature maps for real high-resolution images
        image_features = vgg.predict(high_resolution_images)

        # Train the generator network
        g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],
                                                  [real_labels, image_features])

        # print("g_loss:", g_loss)

        print("Epoch {} : g_loss: {} , d_loss: {}".format(epoch, g_loss[0], d_loss[0]))

        # Write the losses to Tensorboard
        # write_log(tensorboard, 'g_loss', g_loss[0], epoch)
        # write_log(tensorboard, 'd_loss', d_loss[0], epoch)

        # Sample and save images after every 100 epochs

        with writer.as_default():
            tf.summary.scalar('g_loss', g_loss[0], step=epoch)
            tf.summary.scalar('d_loss', d_loss[0], step=epoch)
        writer.flush()

        # write_log(tensorboard, 'g_loss', g_loss[0], epoch)
        # write_log(tensorboard, 'd_loss', d_loss[0], epoch)

        # print("Epoch {}:  Gloss : {} , Dloss {}".format(epoch+1 , g_loss , d_loss))
        if (epoch) % 100 == 0:
            # high_resolution_images, low_resolution_images = sample_images(batch_size)
            # high_resolution_images, low_resolution_images = next(highres_dataloader) , next(lowres_dataloader)
            high_resolution_images, low_resolution_images = next(dataloader)

            # Normalize images
            # high_resolution_images = high_resolution_images / 127.5 - 1.
            # low_resolution_images = low_resolution_images / 127.5 - 1.

            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(res_im_path + "img_{}_{}".format(epoch, index),
                            low_resolution_images[index], generated_images[index], high_resolution_images[index]
                            )
            # Save models
            generator.save_weights(models_path + "generator_{}.h5".format(epoch))
            discriminator.save_weights(models_path + "discriminator_{}.h5".format(epoch))


if __name__ == '__main__':
    pass