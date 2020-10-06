import preprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
from datetime import datetime
import sklearn


class Classifier:
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0, learning_rate=1e-3):
        """Defining the model."""

        self.scope = scope
        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.input = tf.placeholder(tf.float32, [None, img_h, img_w, 1])

        self.conv1 = slim.conv2d(
                self.input,
                num_outputs=32, kernel_size=[3, 8],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv1'
        )
        self.conv2 = slim.conv2d(
                self.conv1,
                num_outputs=64, kernel_size=[5, 5],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv2'
        )
        self.conv3 = slim.conv2d(
                self.conv2,
                num_outputs=128, kernel_size=[5, 5],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv3'
        )
        self.pool = slim.max_pool2d(self.conv3, [2, 2])

        self.hidden = slim.fully_connected(
                slim.flatten(self.pool),
                512,
                scope=self.scope+'_hidden',
                activation_fn=tf.nn.relu
        )
        self.classes = slim.fully_connected(
                tf.nn.dropout(self.hidden, self.dropout_keep_prob),
                self.n_classes,
                scope=self.scope+'_fc',
                activation_fn=None
        )

        self.targets = tf.placeholder(tf.int32, [None])
        self.targets_onehot = tf.one_hot(self.targets, self.n_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.targets_onehot,
                logits=self.classes
        ))
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)


class FineTuningClassifier:
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0, learning_rate=1e-3):
        """Defining the model for fine tuning."""

        self.scope = scope
        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.input = tf.placeholder(tf.float32, [None, img_h, img_w, 1])

        self.conv1 = slim.conv2d(
                self.input,
                num_outputs=32, kernel_size=[3, 8],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv1'
        )
        self.conv2 = slim.conv2d(
                self.conv1,
                num_outputs=64, kernel_size=[5, 5],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv2'
        )
        self.conv3 = slim.conv2d(
                self.conv2,
                num_outputs=128, kernel_size=[5, 5],
                stride=[2, 2], padding='Valid',
                scope=self.scope+'_conv3'
        )
        self.pool = slim.max_pool2d(self.conv3, [2, 2])

        hidden_layer = tf.layers.Dense(
            512,
            activation=tf.nn.relu)
        self.hidden = hidden_layer(slim.flatten(self.pool), scope=self.scope+'_hidden')
        classes_layer = tf.layers.Dense(
            self.n_classes,
            activation=None)
        self.classes = classes_layer(tf.nn.dropout(self.hidden, self.dropout_keep_prob), scope=self.scope+'_fc')

        self.targets = tf.placeholder(tf.int32, [None])
        self.targets_onehot = tf.one_hot(self.targets, self.n_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.targets_onehot,
                logits=self.classes
        ))
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(
                self.loss,
                var_list=[hidden_layer.variables, classes_layer.variables]
        )


class FineTuningClassifierBis:
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0, learning_rate=1e-3):
        """Defining the model for fine tuning."""

        self.scope = scope
        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.model = tf.keras.Sequential()

        self.input = tf.keras.Input(dtype=tf.float32, shape=(None, img_h, img_w, 1))

        conv1_layer = tf.layers.Conv2D(filters=32, kernel_size=[3, 8], stride=[1, 1], padding='valid')
        self.model.add(conv1_layer)

        conv2_layer = tf.layers.Conv2D(filters=64, kernel_size=[5, 5], stride=[2, 2], padding='valid')
        self.model.add(conv2_layer)

        conv3_layer = tf.layers.Conv2D(filters=128, kernel_size=[5, 5], stride=[2, 2], padding='valid')
        self.model.add(conv3_layer)

        pool = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2,2], padding='valid')
        self.model.add(pool)

        self.model.add(tf.layers.Flatten())
        hidden_layer = tf.layers.Dense(units=512, activation=tf.nn.relu)
        self.model.add(hidden_layer)

        self.model.add(tf.layers.Dropout(rate=self.dropout_keep_prob))
        classes_layer = tf.layers.Dense(units=self.n_classes, activation=None)
        self.model.add(classes_layer)

        self.model.layers.conv1_layer.trainable = False
        self.model.layers.conv2_layer.trainable = False
        self.model.layers.conv3_layer.trainable = False

        self.model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate),
                           loss=tf.nn.softmax_cross_entropy_with_logits_v2)


def train(model_name, training_dataset, validation_dataset, train_steps=5e5, pretrained=None, fine_tuning=False,
          learning_rate=1e-3):
    if fine_tuning and pretrained is None:
        raise ValueError("if fine_tuning is True, you must provide a pretrained model.")

    img_h, img_w = 64, 64
    batch_size = 10
    start = datetime.now()
    best_acc = 0
    best_loss = np.infty

    if fine_tuning:
        nn = FineTuningClassifier('classifier', img_w, img_h, len(preprocessing.CLASSES), dropout_keep_prob=0.8,
                                  learning_rate=learning_rate)

        # thanks to https://github.com/KranthiGV/Pretrained-Show-and-Tell-model/issues/7#issuecomment-309862894
        vars_to_rename = {
            "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
            "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
        }
        new_checkpoint_vars = {}
        reader = tf.train.NewCheckpointReader(pretrained)
        for old_name in reader.get_variable_to_shape_map():
            if old_name in vars_to_rename:
                new_name = vars_to_rename[old_name]
            else:
                new_name = old_name
            new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))
        print("new checkpoint vars:", new_checkpoint_vars)
    else:
        nn = Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES), dropout_keep_prob=0.8,
                        learning_rate=learning_rate)

    dataset = list(map(lambda f: f.strip(),
                       open(training_dataset, 'r').readlines()))
    validation_dataset = list(map(lambda f: f.strip(),
                                  open(validation_dataset, 'r').readlines()))

    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)
        if fine_tuning:
            saver_loader = tf.train.Saver(new_checkpoint_vars)
        else:
            saver_loader = tf.train.Saver()
        saver = tf.train.Saver(max_to_keep=3)
        best_acc_saver = tf.train.Saver(max_to_keep=100)
        best_loss_saver = tf.train.Saver(max_to_keep=100)
        summary_writer = tf.summary.FileWriter('summaries/'+model_name)

        # todo debug
        # saver.export_meta_graph("saver2.txt", as_text=True)
        # quit(-1)

        if pretrained is not None:
            saver_loader.restore(sess, pretrained)
            print("pretrained model loaded")

        for t in range(train_steps):
            
            # perform training step
            images, labels = preprocessing.get_batch(dataset, batch_size, (img_h, img_w))
            loss, _ = sess.run([nn.loss, nn.train_step], feed_dict={
                nn.input: images,
                nn.targets: labels
            })

            # show and save training status
            if t % 10000 == 0:
                print("save")
                saver.save(sess, 'saves/step_'+model_name, global_step=t)

            summary = tf.Summary()
            summary.value.add(tag='Loss', simple_value=float(loss))
            if t % 100 == 0:
                # testing model on validation set occasionally
                images, labels = preprocessing.get_batch(
                        validation_dataset, 100, (img_h, img_w))
                classes = sess.run(nn.classes, feed_dict={nn.input: images})
                predictions = np.argmax(classes, -1)

                val_err = float(sum(predictions != labels))
                summary.value.add(tag='ValidationError', simple_value=val_err)

                val_acc = sklearn.metrics.accuracy_score(labels, predictions)
                summary.value.add(tag='ValidationAccuracy', simple_value=val_acc)

                delta = datetime.now() - start
                print(f"step {t}/{train_steps}, loss: {loss}, valErr: {val_err}, valAcc: {val_acc}, elapsed time: {delta}")

                if val_acc > best_acc:
                    best_acc_saver.save(sess, f'saves/best_acc_{model_name}_{best_acc}', global_step=t)
                    best_acc = val_acc
                    print("new val acc", best_acc)

                if loss < best_loss:
                    best_loss_saver.save(sess, f'saves/best_loss_{model_name}_{best_loss}', global_step=t)
                    best_loss = loss
                    print("new loss", best_loss)

            summary_writer.add_summary(summary, t)
            summary_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, required=True, help='Training dataset name')
    parser.add_argument('-v', type=str, required=True, help='Validation dataset name')
    parser.add_argument('-m', type=str, required=True, help='Model name')
    parser.add_argument('-i', type=int, help='Training steps (iterations)')
    parser.add_argument('-l', type=float, help='Learning rate')
    parser.add_argument('-f', action='store_true', help='Perform fine-tuning')
    parser.add_argument('-p', type=str, help='Pretrained model')

    opt = parser.parse_args()

    # es: python train.py -t datasplits/good_train -v datasplits/good_validation -m finetuning
    #       -i 1e6+1 -l 5e-5 -f -p models/top_fnt
    train(opt.m, opt.t, opt.v, train_steps=opt.i, pretrained=opt.p, fine_tuning=opt.f, learning_rate=opt.l)
