import os
import h5py
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_io as tfio
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.model_selection import train_test_split


def plot_3d(image, threshold=0):
    verts, faces, _, _ = measure.marching_cubes(image)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()


def rgb2gray(rgb):
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return gray


def normalization(image):
    std = np.std(image)
    mean = np.mean(image)
    image_normalized = (image - mean) / std
    image_max = np.max(image_normalized)
    image_min = np.min(image_normalized)
    image_normalized = (image_normalized - image_min) / (image_max-image_min)
    return image_normalized * 255


def write_hdf5(origin_images, target_images, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("origin_images", data=origin_images, dtype=origin_images.dtype)
        f.create_dataset("labels", data=target_images, dtype=target_images.dtype)


def load_hdf5(infile, name):
    with h5py.File(infile, "r") as f:
        return f[name][()]


def pre_import_data():
    origin_images = []
    labels = []
    for cur_path, sub_paths, files in os.walk('pancreas'):
        if len(sub_paths) == 0:
            origin_image = np.zeros([512, 512, 60])
            target_image = np.zeros([512, 512, 60])
            for file in files:
                if '.nii' in file.lower():
                    if '.gz' in file.lower():
                        image = np.array(nib.load(f'{cur_path}/{file}').get_fdata())
                        target_image[:, :, 0: image.shape[2]] = image[:, :, :]
                    else:
                        image = np.array(nib.load(f'{cur_path}/{file}').get_fdata())
                        origin_image[:, :, 0: image.shape[2]] = image[:, :, :]
            # origin_image = interpolation.zoom(origin_image, [256 / origin_image.shape[0], 256 / origin_image.shape[1], 32 / origin_image.shape[2]])
            # target_image = interpolation.zoom(target_image, [256 / target_image.shape[0], 256 / target_image.shape[1], 32 / target_image.shape[2]])
            # for index in range(32):
            #     plt.imshow(origin_image[:, :, index], cmap=plt.cm.bone)
            #     plt.show()
            #     plt.imshow(target_image[:, :, index], cmap=plt.cm.bone)
            #     plt.show()
            origin_image = np.transpose(origin_image, [2, 0, 1])
            target_image = np.transpose(target_image, [2, 0, 1])
            for index in range(target_image.shape[0]):
                image = cv2.resize(np.array(origin_image[index, :, :]), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                if image.max() == 0:
                    continue
                image = np.array((np.maximum(image, 0) / image.max()) * 255.0, dtype=np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(image)
                image = np.expand_dims(image, axis=2)
                image = np.concatenate((image, image, image), axis=-1)
                origin_images.append(image)
                if target_image[index, :, :][target_image[index, :, :] == 1].shape[0] == 0:
                    labels.append(0)
                else:
                    labels.append(1)
            # plot_3d(origin_image)
            # plot_3d(target_image)
    origin_images = np.array(origin_images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    labels = np.array(tf.one_hot(labels, 2))
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    x_train, x_test, y_train, y_test = train_test_split(origin_images, labels, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    write_hdf5(x_train, y_train, 'train_classification.hdf5')
    write_hdf5(x_val, y_val, 'val_classification.hdf5')
    write_hdf5(x_test, y_test, 'test_classification.hdf5')


def import_data(data_type):
    images = tfio.IODataset.from_hdf5(f'{data_type}_classification.hdf5', dataset='/origin_images')
    labels = tfio.IODataset.from_hdf5(f'{data_type}_classification.hdf5', dataset='/labels')
    data = tf.data.Dataset.zip((images, labels)).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return data


if __name__ == '__main__':
    pre_import_data()
