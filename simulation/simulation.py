import glob
import os
from PIL import Image
from keras.engine.saving import load_model
import cv2 as cv
import numpy as np

from contours import crop_image
from cnn_model.alex import shape_dict


def begin_simulation(model_path):
    frames = os.scandir("./frames")
    model = load_model(model_path)
    for i, frame in enumerate(frames):
        crop_image(frame.path, "./crops", i)

    crops = os.listdir("./crops")
    x = np.zeros((len(crops), 244, 244, 3), 'float32')
    for i, crop in enumerate(crops):
        x[i] = cv.imread(os.path.join("./crops", crop))/255
    pred = model.predict(np.array(x))
    pred_classes = list(pred.argmax(axis=1))
    pred_probs = [pred[i][pred_classes[i]] for i in range(len(pred))]
    print(pred_classes, "\n", pred_probs)

    frames = os.scandir("./frames")
    for i, frame in enumerate(frames):
        Image.open(frame.path).convert('RGBA').show()

        crops = glob.glob("./crops/crop_" + str(i) + "_*.png")
        for crop in crops:
            p = pred_probs.pop(0)
            if p > 0.7:
                print(crop.split("_")[2], list(shape_dict.keys())[pred_classes.pop(0)], p)
                Image.open(crop).convert('RGBA').show()


if __name__ == '__main__':
    begin_simulation("../cnn_model/models/alex/alex_shapes_2.h5")
