# from __future__ import print_function
import fnmatch
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

shape_dict = {'circle': 0, 'semicircle': 1, 'quartercircle': 2, 'triangle': 3, 'square': 4, 'rectangle': 5,
              'trapezoid': 6, 'pentagon': 7, 'hexagon': 8, 'heptagon': 9, 'octagon': 10, 'star': 11, 'cross': 12}


def preprocessing_gray(dirpath):
    print('START PREPROCESSING')

    num_images = len(fnmatch.filter(os.listdir(dirpath), '*.png'))
    input_images = os.scandir(dirpath)
    x = np.zeros((num_images, 244, 244), 'float32')
    y = np.zeros(num_images)

    for i, image_entry in enumerate(input_images):
        print(image_entry.name + " --- " + str(100 * i / num_images) + "%")

        image = cv2.imread(image_entry.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x[i] = image
        shape_name, _ = os.path.splitext(image_entry.name)
        shape_name = shape_name.split("_")[0]
        y[i] = shape_dict[shape_name]

        # cv2.imwrite('out_img_gray/' + image_entry.name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print(f'X: {x.shape}')
    print(f'Y shape: {y.shape}')
    print(f'Y: {np.unique(y)}')
    print('END PREPROCESSING')

    return x, y


def preprocessing_bw(dirpath):
    print('START PREPROCESSING')

    num_images = len(fnmatch.filter(os.listdir(dirpath), '*.png'))
    input_images = os.scandir(dirpath)
    x = np.zeros((num_images, 244, 244), 'float32')
    y = np.zeros(num_images)

    for i, image_entry in enumerate(input_images):
        print(image_entry.name + " --- " + str(100 * i / num_images) + "%")

        image = cv2.imread(image_entry.path)

        # denoising
        # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # binary
        # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # negative
        hist = cv2.calcHist([image], [0], None, [2], [0, 256]).ravel()
        if hist[0] < hist[1]:
            image = cv2.bitwise_not(image)

        # erosion
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)

        # dilation
        image = cv2.dilate(image, kernel, iterations=1)

        x[i] = image
        shape_name, _ = os.path.splitext(image_entry.name)
        shape_name = shape_name.split("_")[0]
        y[i] = shape_dict[shape_name]

        # cv2.imwrite('out_img_ben/' + image_entry.name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print(f'X: {x.shape}')
    print(f'Y shape: {y.shape}')
    print(f'Y: {np.unique(y)}')
    print('END PREPROCESSING')

    return x, y


def cnn(X, Y, name):
    print('START CNN')

    checkpoint_path = "models/cp_" + name + ".ckpt"  # https://www.tensorflow.org/tutorials/keras/save_and_load
    checkpoint_dir = os.path.dirname(checkpoint_path)
    batch_size = 128
    num_classes = 13
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 244, 244

    # the data, split between train and test sets
    x_train, x_test_val, y_train, y_test_val = train_test_split(X, Y, test_size=0.4, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=42,
                                                    shuffle=True)
    del x_test_val, y_test_val

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train/255
    x_test = x_test/255
    x_val = x_val/255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_val.shape[0], 'validation samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[cp_callback])

    model.save_weights("models/weights_" + name)
    model.save("models/" + name + ".h5")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    # path = "../dataset_generation/shapes_generation/out_img/"
    path = "../dataset_generation/shapes_generation/out_bw_img/"

    x1, y1 = preprocessing_gray(path)
    cnn(x1, y1, "cnn_simple")   # grey scale

    # x2, y2 = preprocessing_bw(path)
    # cnn(x2, y2, "cnn_bw")  # binarization
