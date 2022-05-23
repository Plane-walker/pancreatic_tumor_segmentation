import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from import_data_classification import import_data, load_hdf5
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def identity_block(inputs, filters):
    hidden = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    hidden = tf.keras.layers.BatchNormalization(axis=3)(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(hidden)
    hidden = tf.keras.layers.BatchNormalization(axis=3)(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Add()([hidden, inputs])
    outputs = tf.keras.layers.Activation('relu')(hidden)
    outputs = tf.keras.layers.Dropout(0.1)(outputs)
    return outputs


def conv_block(inputs, filters):
    hidden = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(inputs)
    hidden = tf.keras.layers.BatchNormalization(axis=3)(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(hidden)
    hidden = tf.keras.layers.BatchNormalization(axis=3)(hidden)
    inputs_skip = tf.keras.layers.Conv2D(filters, (1, 1), strides=(2, 2))(inputs)
    hidden = tf.keras.layers.Add()([hidden, inputs_skip])
    outputs = tf.keras.layers.Activation('relu')(hidden)
    outputs = tf.keras.layers.Dropout(0.1)(outputs)
    return outputs


def get_model(image_shape, filter_size, classes):
    inputs = tf.keras.layers.Input(shape=image_shape)
    hidden = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=tf.keras.layers.Input(shape=image_shape))(inputs)
    # hidden = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
    # hidden = tf.keras.layers.Conv2D(32, kernel_size=7, strides=2, padding='same')(hidden)
    # hidden = tf.keras.layers.BatchNormalization()(hidden)
    # hidden = tf.keras.layers.Activation('relu')(hidden)
    # hidden = tf.keras.layers.Dropout(0.1)(hidden)
    # hidden = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(hidden)
    # block_layers = [3, 4, 3]
    # for i in range(3):
    #     if i == 0:
    #         for j in range(block_layers[i]):
    #             hidden = identity_block(hidden, filter_size)
    #     else:
    #         filter_size = filter_size * 2
    #         hidden = conv_block(hidden, filter_size)
    #         for j in range(block_layers[i] - 1):
    #             hidden = identity_block(hidden, filter_size)
    hidden = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(hidden)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def train():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu0 = gpus[0]
        tf.config.experimental.set_memory_growth(gpu0, True)
        tf.config.set_visible_devices([gpu0], "GPU")
    train_data = import_data('train')
    val_data = import_data('val')
    epochs = 15
    lr = 1e-3
    decay_rate = lr / epochs
    model = get_model((256, 256, 3), 32, 2)
    sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.8, decay=decay_rate, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    json_string = model.to_json()
    open('model_architecture_classification.json', 'w').write(json_string)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights_classification.h5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=20,
                                                      verbose=1,
                                                      mode='min')
    history = model.fit(train_data,
                        epochs=epochs,
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
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    fig.set_tight_layout(True)
    plt.savefig("train_result_classification.png")
    plt.show()
# plt.imshow(images[10] * 255, cmap=plt.cm.bone)
# plt.show()


def test():
    images_test = load_hdf5('test_classification.hdf5', 'origin_images')
    labels_test_int = tf.argmax(load_hdf5('test_classification.hdf5', 'labels'), axis=1)
    with open(r'model_architecture_classification.json', 'r') as file:
        model_json = file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('best_weights_classification.h5')
    predictions = model.predict(images_test, batch_size=64, verbose=2)
    predictions_int = tf.argmax(predictions, axis=1)
    cm = confusion_matrix(labels_test_int, predictions_int)
    accuracy = accuracy_score(labels_test_int, predictions_int)
    recall = recall_score(labels_test_int, predictions_int, average='weighted')
    precision = precision_score(labels_test_int, predictions_int, average='weighted')
    f1 = f1_score(labels_test_int, predictions_int, average='weighted')

    print(cm)
    print("accuracy: {:.4f}".format(accuracy))
    print("recall: {:.4f}".format(recall))
    print("precision: {:.4f}".format(precision))
    print("F1: {:.4f}".format(f1))
    precision, recall, fscore, support = precision_recall_fscore_support(labels_test_int, predictions_int)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


if __name__ == '__main__':
    # train()
    test()
