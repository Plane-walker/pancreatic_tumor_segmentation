import os
import tensorflow as tf
import cv2
import imageio
import matplotlib.image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from import_data import load_hdf5

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

def png2gif(filelist, name, duration=0.5):
    print(filelist)
    a = matplotlib.image.imread('./result/0.png')
    print(a)
    print(a.dtype)
    frames = []
    for i in range(32):
        frames.append(matplotlib.image.imread('./result/'+str(i)+'.png'))
    imageio.mimsave(name, frames, 'GIF', duration=0.3)

def result_plot():
    origin_images = []
    target_images = []
    num = 0
    for cur_path, sub_paths, files in os.walk('pancreas'):
        num += 1
        if num == 100:
            break
        print(num)
        if num == 85:
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
                origin_image = np.transpose(origin_image, [2, 0, 1])
                target_image = np.transpose(target_image, [2, 0, 1])
                for index in range(target_image.shape[0]):
                    print(index)
                    if target_image[index, :, :][target_image[index, :, :] == 1].shape[0] == 0:
                        continue
                    image = cv2.resize(np.array(origin_image[index, :, :]), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                    image = np.array((np.maximum(image, 0) / image.max()) * 255.0, dtype=np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    image = clahe.apply(image)
                    image = np.expand_dims(image, axis=-1)
                    origin_images.append(image)
                    image = cv2.resize(np.array(target_image[index, :, :]), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                    image[image > 0.5] = 1
                    image[image < 0.5] = 0
                    image = np.expand_dims(image, axis=-1)
                    target_images.append(image)
            # origin_image[target_image > 0.5] == 255
            # for index in range(32):
            #     plt.imshow(origin_image[:, :, index], cmap=plt.cm.bone)
            #     plt.show()
            #     plt.imshow(target_image[:, :, index], cmap=plt.cm.bone)
            #     plt.show()
            # plot_3d(origin_image)
            # plot_3d(target_image)
    origin_images = np.array(origin_images)
    target_images = np.array(target_images)
    origin_images[target_images > 0.5] = 255
    # print(origin_images.shape)
    # print(target_images.shape)

    for index in range(origin_images.shape[0]):
        plt.imshow(origin_images[index, :, :], cmap=plt.cm.bone)
        plt.savefig('./result_85/' + str(index) + '.png')

    origin_images = load_hdf5('test.hdf5', 'origin_images')
    print(origin_images.shape)
    with open(r'model_architecture.json', 'r') as file:
        model_json = file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('best_weights.h5')
    predictions = model.predict(origin_images, batch_size=4, verbose=2)
    origin_images[predictions > 0.5] = 255
    for index in range(origin_images.shape[0]):
        plt.imshow(origin_images[index, :, :], cmap=plt.cm.bone)
        plt.savefig('./test_result/' + str(index) + '.png')


    origin_images = load_hdf5('val.hdf5', 'origin_images')
    print(origin_images.shape)
    predictions = model.predict(origin_images, batch_size=4, verbose=2)
    origin_images[predictions > 0.5] = 255
    for index in range(origin_images.shape[0]):
        plt.imshow(origin_images[index, :, :], cmap=plt.cm.bone)
        plt.savefig('./val_result/' + str(index) + '.png')

    origin_images = load_hdf5('train.hdf5', 'origin_images')
    print(origin_images.shape)
    predictions = model.predict(origin_images, batch_size=4, verbose=2)
    origin_images[predictions > 0.5] = 255
    for index in range(origin_images.shape[0]):
        plt.imshow(origin_images[index, :, :], cmap=plt.cm.bone)
        plt.savefig('./train_result/' + str(index) + '.png')

def compare():
    for i in range(54):
        plt.subplot(1, 2, 1)
        plt.imshow(matplotlib.image.imread('./result_85/'+str(i)+'.png'), cmap=plt.cm.bone)
        plt.title('Target Result No.'+str(i+1))
        plt.subplot(1, 2, 2)
        plt.imshow(matplotlib.image.imread('./85_pred/'+str(i)+'.png'), cmap=plt.cm.bone)
        plt.title('Prediction Result No.'+str(i+1))
        plt.savefig('./compare_result/' + str(i) + '.png')
        print(i)
    frames = []
    for i in range(54):
        frames.append(matplotlib.image.imread('./compare_result/'+str(i)+'.png'))
    imageio.mimsave('85', frames, 'GIF', duration=0.3)


if __name__ == '__main__':
    result_plot()
    list = os.listdir('./result/')
    png2gif(list, 'name')
    compare()