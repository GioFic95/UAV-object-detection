from __future__ import print_function
import glob
import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

shape_dict = {'circle': 0, 'semicircle': 1, 'quartercircle': 2, 'triangle': 3, 'square': 4, 'rectangle': 5,
              'trapezoid': 6, 'pentagon': 7, 'hexagon': 8, 'heptagon': 9, 'octagon': 10, 'star': 13, 'cross': 12}


def preprocessing_gray():
    print('START PREPROCESSING')

    input_images = glob.glob("../dataset_generation/shapes_generation/out_img/*.png")
    x = []
    y = []
    id = 0

    for image_name in input_images:
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x += [image]
        file_name = os.path.basename(image_name)
        shape_name, _ = os.path.splitext(os.path.basename(image_name))
        shape_name = shape_name.split("_")[0]
        y += [shape_dict[shape_name]]

        cv2.imwrite('out_img_gray/'+file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        id += 1

    print(f'X: {np.asarray(x).shape}')
    print(f'Y shape: {np.asarray(y).shape}')
    print(f'Y: {np.unique(np.asarray(y))}')
    print('END PREPROCESSING')

    return np.asarray(x), np.asarray(y)


def preprocessing_bw():
    print('START PREPROCESSING')

    input_images = glob.glob("../dataset_generation/shapes_generation/out_img/*.png")
    x = []
    y = []
    id = 0

    for image_name in input_images:
        image = cv2.imread(image_name)

        # denoising
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

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

        x += [image]

        file_name = os.path.basename(image_name)
        shape_name, _ = os.path.splitext(os.path.basename(image_name))
        shape_name = shape_name.split("_")[0]
        y += [shape_dict[shape_name]]

        cv2.imwrite('out_img_ben/'+file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        id += 1

    print(f'X: {np.asarray(x).shape}')
    print(f'Y shape: {np.asarray(y).shape}')
    print(f'Y: {np.unique(np.asarray(y))}')
    print('END PREPROCESSING')

    return np.asarray(x), np.asarray(y)


def cnn(X, Y):
    print('START CNN')

    batch_size = 128
    num_classes = 13
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 244, 244

    # the data, split between train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    x, y = preprocessing_gray()
    cnn(x, y)