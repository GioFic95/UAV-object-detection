import os

import plotly.graph_objects as go
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix as cm1
from tensorflow import confusion_matrix as cm2
from keras.engine.saving import load_model
import numpy as np
from alex import preprocessing, char_dict, shape_dict


def plot_acc(in_path, out_path):
    df = pd.read_csv(in_path)
    print(df)

    fig = go.Figure()

    # Create traces
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_acc'], name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_acc'], name='Validation Accuracy'))

    # Edit the layout
    fig.update_layout(title='',
                      xaxis_title='Epochs',
                      yaxis_title='Score')

    fig.show()

    name = os.path.splitext(os.path.basename(in_path))[0] + "_acc"
    fig.write_image(os.path.join(out_path, name + ".png"))


def plot_loss(in_path, out_path):
    df = pd.read_csv(in_path)
    print(df)

    fig = go.Figure()

    # Create traces
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], name='Training Loss'))
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], name='Validation Loss'))

    # Edit the layout
    fig.update_layout(title='',
                      xaxis_title='Epochs',
                      yaxis_title='Score')

    fig.show()

    name = os.path.splitext(os.path.basename(in_path))[0] + "_loss"
    fig.write_image(os.path.join(out_path, name + ".png"))


def print_confusion_matrix(model_path, input_path, shapes):
    print("confusion matrix")
    num_classes = 13 if shapes else 36

    x, y = preprocessing(input_path, shapes)
    print(y)

    x = x[:10000]
    y = y[:10000]
    print(x.shape)

    model = load_model(model_path)
    print("loaded")

    y_pred = model.predict(x)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred, y_pred.shape)

    y_test = to_categorical(y, num_classes)
    score = model.evaluate(x, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('Confusion Matrix')
    scikit_matrix = cm1(y, y_pred)
    print(scikit_matrix)
    tf_matrix = cm2(y, y_pred, num_classes)
    print(tf_matrix)
    print(tf_matrix == scikit_matrix)
    print('Classification Report')
    target_names = list(shape_dict.keys()) if shapes else list(char_dict.keys())
    print(classification_report(y, y_pred, target_names=target_names))


if __name__ == '__main__':
    plot_path = "./plots"
    # plot_acc_loss("./models/alex/char_2_log.txt", plot_path)
    # plot_acc_loss("./models/alex/char_1_log.txt", plot_path)
    # plot_acc("./models/alex/alex_char_3_log.csv", plot_path)
    # plot_loss("./models/alex/alex_char_3_log.csv", plot_path)
    # plot_acc("./models/alex/log_alex_shapes_2.csv", plot_path)
    # plot_loss("./models/alex/log_alex_shapes_2.csv", plot_path)
    # print_confusion_matrix("./models/alex/alex_char_3.h5",
    #                        "../dataset_generation/chars_generation/out_img_0", False)
    print_confusion_matrix("./models/alex/cp_alex_shapes_2_0030_0.96.ckpt",
                           "../dataset_generation/shapes_generation/out_img", True)
