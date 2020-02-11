import glob
import os
import cv2
import numpy as np


def preprocessing():
    input_images = glob.glob("../dataset_generation/shapes_generation/out_img/*.png")
    x = []
    y = []
    id = 0

    for image_name in input_images:
        image = cv2.imread(image_name)

        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # binary
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # negative
        hist = cv2.calcHist([thresh], [0], None, [2], [0, 256]).ravel()
        if hist[0] < hist[1]:
            thresh = cv2.bitwise_not(thresh)

        # erosion
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(thresh, kernel, iterations=1)
        # dilation
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

        x += [img_dilation]

        shape_name, _ = os.path.splitext(os.path.basename(image_name))
        shape_name = shape_name.split("_")[0]
        y += [shape_name]

        cv2.imwrite('out_img/test_'+str(id)+'.png', img_dilation)
        id += 1

    return x, y


"""
def cnn(x, y):
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
"""

if __name__ == '__main__':
    x, y = preprocessing()
    # cnn(x, y)
