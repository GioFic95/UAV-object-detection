import glob
import os
from keras.engine.saving import load_model
import cv2 as cv
import numpy as np

from contours import crop_image
from cnn_model.alex import shape_dict


def begin_simulation(model_path):
    frames = os.scandir("./frames")
    model = load_model(model_path)
    font = cv.FONT_HERSHEY_SIMPLEX
    rois = []
    for i, frame in enumerate(frames):
        rois += crop_image(frame.path, "./crops", i)

    crops = os.listdir("./crops")
    x = np.zeros((len(crops), 244, 244, 3), 'float32')
    for i, crop in enumerate(crops):
        x[i] = cv.imread(os.path.join("./crops", crop))/255
    pred = model.predict(np.array(x))
    pred_classes = list(pred.argmax(axis=1))
    pred_probs = [pred[i][pred_classes[i]] for i in range(len(pred))]
    print(pred_classes, "\n", pred_probs)

    frames = os.scandir("./frames")
    print("tot num crops:", len(os.listdir("./crops")))
    assert len(os.listdir("./crops")) == len(pred_probs) == len(pred_classes)

    for i, frame in enumerate(frames):
        cv.destroyAllWindows()
        crops = glob.glob("./crops/crop_" + str(i) + "_*.png")
        for crop in crops:
            p = pred_probs.pop(0)
            c = list(shape_dict.keys())[pred_classes.pop(0)]
            r = rois.pop(0)
            print(r)
            if p > 0.9:
                text = os.path.basename(crop) + ": class '" + str(c) + "' with probability {:.3f}".format(p)
                print(text)
                # Image.open(crop).convert('RGBA').show()
                window_name = 'Frame ' + str(i)
                cv.namedWindow(window_name, cv.WINDOW_NORMAL)
                img = cv.imread(frame.path)
                img = cv.rectangle(img, r[0], r[1], (255, 0, 0), 3)
                cv.putText(img, text, (100, img.shape[0] - 200), font, 1, (0, 0, 0), 2, cv.LINE_AA)
                cv.resizeWindow(window_name, 1000, 666)
                cv.imshow(window_name, img)
                k = cv.waitKey(0) & 0xFF
                if k == 27:  # wait for ESC key to go to next image
                    cv.destroyAllWindows()


if __name__ == '__main__':
    begin_simulation("../cnn_model/models/alex/alex_shapes_2.h5")
