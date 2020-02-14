import fnmatch

import keras
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.python.training.saver import latest_checkpoint
from keras.engine.saving import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D


data_path = "../dataset_generation/shapes_generation/out_img/"
array_path = "./arrays/"
batch_size = 128
num_classes = 13
epochs = 10
np.random.seed(1000)
img_rows, img_cols = 244, 244   # input image dimensions
shape_dict = {'circle': 0, 'semicircle': 1, 'quartercircle': 2, 'triangle': 3, 'square': 4, 'rectangle': 5,
              'trapezoid': 6, 'pentagon': 7, 'hexagon': 8, 'heptagon': 9, 'octagon': 10, 'star': 11, 'cross': 12}


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


def preprocessing(dirpath):
    print('START PREPROCESSING')

    num_images = len(fnmatch.filter(os.listdir(dirpath), '*.png'))
    input_images = os.scandir(dirpath)
    x = np.zeros((num_images, img_rows, img_cols), 'float32')
    y = np.zeros(num_images)

    for i, image_entry in enumerate(input_images):
        print(image_entry.name + " --- " + str(100 * i / num_images) + "%")

        image = cv2.imread(image_entry.path)

        x[i] = image
        shape_name, _ = os.path.splitext(image_entry.name)
        shape_name = shape_name.split("_")[0]
        y[i] = shape_dict[shape_name]

    print(f'X: {x.shape}')
    print(f'Y shape: {y.shape}')
    print(f'Y: {np.unique(y)}')
    print('END PREPROCESSING')

    x = x.reshape(x.shape[0], img_rows, img_cols, 3)
    print(f'X: {x.shape}')
    x /= 255

    return x, y


def alex(X, Y, name, epochs=10, load_checkpoint=False):
    checkpoint_path = "models/alex/cp_" + name + "_{epoch:04d}.ckpt"  # https://www.tensorflow.org/tutorials/keras/save_and_load
    checkpoint_dir = os.path.dirname(checkpoint_path)

    print('START ALEX')

    # the data, split between train and test sets
    x_train, x_test_val, y_train, y_test_val = train_test_split(X, Y, test_size=0.4, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=42, shuffle=True)
    del x_test_val, y_test_val

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
        # Instantiate an empty model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(img_rows, img_cols, 3), kernel_size=(11, 11), strides=(4, 4),
                         padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(img_rows * img_cols * 3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(num_classes))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(17))
        model.add(Activation('softmax'))

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor='val_accuracy',
                                                  verbose=1,
                                                  mode='max')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam,
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
    X, Y = preprocessing("../dataset_generation/shapes_generation/sample/")
    alex(X, Y, "alex_1")
