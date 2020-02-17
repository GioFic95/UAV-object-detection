import os

import plotly.graph_objects as go
import pandas as pd


def plot_acc_loss(in_path, out_path):
    df = pd.read_csv(in_path)
    print(df)

    fig = go.Figure()

    # Create traces
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], name='Training Loss'))
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_acc'], name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], name='Validation Loss'))
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_acc'], name='Validation Accuracy'))

    # Edit the layout
    fig.update_layout(title='',
                      xaxis_title='Epochs',
                      yaxis_title='Score')

    fig.show()

    name = os.path.splitext(os.path.basename(in_path))[0]
    fig.write_image(os.path.join(out_path, name + ".png"))


if __name__ == '__main__':
    plot_path = "./plots"
    plot_acc_loss("./models/alex/char_2_log.txt", plot_path)
    plot_acc_loss("./models/alex/char_1_log.txt", plot_path)
