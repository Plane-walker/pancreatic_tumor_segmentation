from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, concatenate, UpSampling3D
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from import_data import import_data, load_hdf5


def res_block(x, nb_filters, strides):
    res_path = tf.keras.layers.BatchNormalization()(x)
    res_path = tf.keras.layers.Activation(activation='relu')(res_path)
    res_path = tf.keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = tf.keras.layers.BatchNormalization()(res_path)
    res_path = tf.keras.layers.Activation(activation='relu')(res_path)
    res_path = tf.keras.layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)
    shortcut = tf.keras.layers.Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    res_path = tf.keras.layers.add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []
    main_path = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = tf.keras.layers.BatchNormalization()(main_path)
    main_path = tf.keras.layers.Activation(activation='relu')(main_path)
    main_path = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)
    shortcut = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    main_path = tf.keras.layers.add([shortcut, main_path])
    to_decoder.append(main_path)
    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    return to_decoder


def decoder(x, from_encoder):
    main_path = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])
    main_path = tf.keras.layers.UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])
    main_path = tf.keras.layers.UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])
    return main_path


def res_u_net(inputs):
    to_decoder = encoder(inputs)
    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])
    path = decoder(path, from_encoder=to_decoder)
    path = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)
    return Model(inputs=inputs, outputs=path)


def u_net_3d(inputs):
    x = inputs
    conv1 = Conv3D(8, 3, activation='relu', padding='same', data_format="channels_last")(x)
    conv1 = Conv3D(8, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(16, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(16, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(32, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(32, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(64, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv3D(64, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(128, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv3D(128, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(64, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv3D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv3D(64, 3, activation='relu', padding='same')(conv6)

    up7 = Conv3D(32, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv3D(32, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(32, 3, activation='relu', padding='same')(conv7)

    up8 = Conv3D(16, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv3D(16, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv3D(16, 3, activation='relu', padding='same')(conv8)

    up9 = Conv3D(8, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv3D(8, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv3D(8, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model


def dice_coe(y_true, y_pred, smooth=1.):
    y_true_f = tf.reshape(y_true, [-1])
    y_true_f = tf.cast(y_true_f, 'float32')
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)
    return (2. * intersection + smooth) / (union + smooth)


def jaccard(y_true, y_pred, smooth=1.):
    y_true_f = tf.reshape(y_true, [-1])
    y_true_f = tf.cast(y_true_f, 'float32')
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)
    return (intersection + smooth) / (union - intersection + smooth)


def dice_loss(y_true, y_pred, smooth=1.):
    return 1 - dice_coe(y_true, y_pred, smooth)


def train():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu0 = gpus[0]
        tf.config.experimental.set_memory_growth(gpu0, True)
        tf.config.set_visible_devices([gpu0], "GPU")
    train_data = import_data('train')
    val_data = import_data('val')
    model = res_u_net(tf.keras.layers.Input(shape=(256, 256, 1)))
    model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=[dice_loss], metrics=['accuracy', dice_coe])
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights.h5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      verbose=1,
                                                      mode='min')
    history = model.fit(train_data,
                        epochs=150,
                        shuffle=True,
                        validation_data=val_data,
                        callbacks=[checkpoint, early_stopping])

    fig = plt.figure(figsize=(6.2, 4.8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Dice Loss')
    plt.plot(history.history['val_loss'], label='Validation Dice Loss')
    plt.title('Training and Validation Dice Loss')
    plt.legend()
    fig.set_tight_layout(True)
    plt.savefig("train_result.png")
    plt.show()


def test():
    origin_images = load_hdf5('test.hdf5', 'origin_images')
    origin_images_bak = load_hdf5('test.hdf5', 'origin_images')
    target_images = load_hdf5('test.hdf5', 'target_images')
    with open(r'model_architecture.json', 'r') as file:
        model_json = file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('best_weights.h5')
    predictions = model.predict(origin_images, batch_size=4, verbose=2)
    # origin_images[predictions > 0.5] = 255
    # origin_images_bak[target_images > 0.5] = 255
    # for index in range(origin_images.shape[0]):
    #     plt.imshow(origin_images_bak[index, :, :], cmap=plt.cm.bone)
    #     plt.title('0')
    #     plt.show()
    #     plt.imshow(origin_images[index, :, :], cmap=plt.cm.bone)
    #     plt.title('1')
    #     plt.show()
    dice_index = dice_coe(target_images, predictions)
    print(f"dice: {dice_index}")
    hausdorff_index = 0
    for index in range(target_images.shape[0]):
        _, target_image = cv2.threshold(target_images[index, :, :], 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(target_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        target_image = np.array(contours[0])
        target_image = np.reshape(target_image, [target_image.shape[0], target_image.shape[2]])
        _, prediction = cv2.threshold(predictions[index, :, :], 0.5, 255, cv2.THRESH_BINARY)
        prediction = np.array(prediction, dtype=np.uint8)
        contours, _ = cv2.findContours(prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            prediction = np.array([[(0, 0)]])
        else:
            prediction = np.array(contours[0])
        prediction = np.reshape(prediction, [prediction.shape[0], prediction.shape[2]])
        hausdorff_index += directed_hausdorff(target_image, prediction)[0] / target_images.shape[0]
    print(f"hausdorff: {hausdorff_index}")
    jaccard_index = jaccard(target_images, predictions)
    print(f"jaccard: {jaccard_index}")


if __name__ == '__main__':
    # train()
    test()
