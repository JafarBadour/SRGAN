import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import tensorflow as tf

def input_pipeline(data_path, batch_size, highres_shape, lowres_shape):
    all_images = glob.glob(data_path + "*")
    print("######",all_images)
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

    fig = plt.figure(figsize=(12, 4))

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
