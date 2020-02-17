import fnmatch
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.python.training.saver import latest_checkpoint
import keras
from keras.engine.saving import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization


sample_path = "../dataset_generation/shapes_generation/sample/"
data_path = "../dataset_generation/shapes_generation/out_img/"
chars_path = "../dataset_generation/chars_generation/out_img/"
array_path = "./arrays/"
models_path = "./models/alex/"
batch_size = 128
np.random.seed(1000)
img_rows, img_cols = 224, 224   # input image dimensions
shape_dict = {'circle': 0, 'semicircle': 1, 'quartercircle': 2, 'triangle': 3, 'square': 4, 'rectangle': 5,
              'trapezoid': 6, 'pentagon': 7, 'hexagon': 8, 'heptagon': 9, 'octagon': 10, 'star': 11, 'cross': 12}
char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
             'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31,
             'W': 32, 'X': 33, 'Y': 34, 'Z': 35}


def serialized_preprocessing(iterable):
    print('START PREPROCESSING')
    images, proc_name = iterable
    x = np.zeros((len(images), img_rows, img_cols, 3), 'float32')
    y = np.zeros(len(images))

    for i, image_entry in enumerate(images):
        print(proc_name + " ->" + image_entry + " --- " + str(100 * i / len(images)) + "%")

        image = cv2.imread(image_entry)

        x[i] = image
        shape_name, _ = os.path.splitext(image_entry)
        shape_name = shape_name.split("_")[0]
        y[i] = shape_dict[shape_name]

    print(f'X: {x.shape}')
    print(f'Y shape: {y.shape}')
    print(f'Y: {np.unique(y)}')
    print('END PREPROCESSING')

    # serialization
    with open(array_path + proc_name + ".npy", 'wb') as f:
        np.save(f, x)
        np.save(f, y)


def parallel_serialized_preprocessing():
    cpu_num = multiprocessing.cpu_count()
    input_images = os.listdir(data_path)
    x_in = np.array_split(input_images, cpu_num)
    names = ["proc_" + str(n) for n in range(cpu_num)]
    print(x_in)

    if __name__ == '__main__':
        with multiprocessing.Pool(processes=cpu_num) as p:
            p.map(serialized_preprocessing, zip(x_in, names))


def deserialization():
    arrays = os.listdir("./arrays/")
    X = np.zeros((0, img_rows, img_cols, 3), 'float32')
    Y = np.zeros((0), 'float32')

    for i in range(len(arrays)):
        with open("./arrays/" + arrays[i], 'rb') as f1:
            x1 = np.load(f1)
            y1 = np.load(f1)
            print(i, x1.shape, y1.shape)
            X = np.concatenate((X, x1))
            Y = np.concatenate((Y, y1))

    print(X.shape)
    print(Y.shape)


def preprocessing(dirpath, shapes):
    print('START PREPROCESSING')

    num_images = len(fnmatch.filter(os.listdir(dirpath), '*.png'))
    input_images = os.scandir(dirpath)
    x = np.zeros((num_images, img_rows, img_cols, 3), 'float32')
    y = np.zeros(num_images)

    for i, image_entry in enumerate(input_images):
        print(image_entry.name + " --- " + str(100 * i / num_images) + "%")

        image = cv2.imread(image_entry.path)
        if image.shape[0] != img_rows or image.shape[1] != img_cols:
            image.resize((img_rows, img_cols, image.shape[2]))
        x[i] = image

        img_name, _ = os.path.splitext(image_entry.name)
        shape_name = img_name.split("_")[0]
        char_name = img_name.split("_")[0]
        y[i] = shape_dict[shape_name] if shapes else char_dict[char_name]

        # cv2.imwrite("./out_img_resize/" + image_entry.name, image)

    print(f'X: {x.shape}')
    print(f'Y shape: {y.shape}')
    print(f'Y: {np.unique(y)}')
    print('END PREPROCESSING 1')

    # normalize x
    x /= 255
    print('END PREPROCESSING 2')

    # shuffle x and y
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation]
    y = y[permutation]
    del permutation
    print('END PREPROCESSING 3')
    return x, y


def train_test_val(X, Y, train_proportion, test_proportion):
    train_ratio = int(X.shape[0] * train_proportion)
    test_ratio = train_ratio + int(X.shape[0] * test_proportion)
    X_train = X[:train_ratio, :]
    X_test = X[train_ratio:test_ratio, :]
    X_val = X[test_ratio:, :]
    Y_train = Y[:train_ratio]
    Y_test = Y[train_ratio:test_ratio]
    Y_val = Y[test_ratio:]
    return X_train, X_test, X_val, Y_train, Y_test, Y_val


def alex(X, Y, name, epochs, num_classes, load_checkpoint=False):
    checkpoint_path = models_path + "cp_" + name + "_{epoch:04d}_{val_acc:.2f}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    print('START ALEX')

    # the data, split between train and test sets
    x_train, x_test, x_val, y_train, y_test, y_val = train_test_val(X, Y, 0.6, 0.2)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_val.shape[0], 'validation samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    y_val = to_categorical(y_val, num_classes)

    if load_checkpoint:
        latest = latest_checkpoint(checkpoint_dir)
        model = load_model(latest)
    else:
        # Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(img_rows, img_cols, 3), kernel_size=(11, 11),
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling 
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(img_rows * img_cols * 3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor='val_acc',
                                                  verbose=1,
                                                  mode='max')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[cp_callback])

    model.save_weights(models_path + "weights_" + name)
    model.save(models_path + name + ".h5")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    # training on shapes
    # X, Y = preprocessing(data_path, shapes=True)
    # alex(X, Y, "alex_shapes_1", 30, 13)

    # training on chars
    X, Y = preprocessing(chars_path, shapes=False)
    # alex(X, Y, "alex_char_2", 30, 36)

