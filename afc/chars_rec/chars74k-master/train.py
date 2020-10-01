import preprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
from datetime import datetime
import sklearn


class Classifier:
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0):
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
        self.train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)


def train(model_name, training_dataset, validation_dataset, train_steps):
    img_h, img_w = 64, 64
    batch_size = 10
    start = datetime.now()

    nn = Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES), 0.8)
    dataset = list(map(lambda f: f.strip(),
                       open(training_dataset, 'r').readlines()))
    validation_dataset = list(map(lambda f: f.strip(),
                                  open(validation_dataset, 'r').readlines()))

    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('summaries/'+model_name)

        for t in range(train_steps):
            
            # perform training step
            images, labels = preprocessing.get_batch(dataset, batch_size, (img_h, img_w))
            loss, _ = sess.run([nn.loss, nn.train_step], feed_dict={
                nn.input: images,
                nn.targets: labels
            })

            # show and save training status
            if t % 10000 == 0:
                saver.save(sess, 'saves/'+model_name, global_step=t)

            summary = tf.Summary()
            summary.value.add(tag='Loss', simple_value=float(loss))
            if t % 100 == 0:
                # testing model on validation set occasionally
                images, labels = preprocessing.get_batch(
                        validation_dataset, 20, (img_h, img_w))
                classes = sess.run(nn.classes, feed_dict={nn.input: images})
                predictions = np.argmax(classes, -1)

                val_err = float(sum(predictions != labels))
                summary.value.add(tag='ValidationError', simple_value=val_err)

                val_acc = sklearn.metrics.accuracy_score(labels, predictions)
                summary.value.add(tag='ValidationAccuracy', simple_value=val_acc)

                delta = datetime.now() - start
                print(f"step {t}/{train_steps}, loss: {loss}, valErr: {val_err}, valAcc: {val_acc}, elapsed time: {delta}")
            summary_writer.add_summary(summary, t)
            summary_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-t', type=str, required=True, help='Training dataset name')
    parser.add_argument(
            '-v', type=str, required=True, help='Validation dataset name')
    parser.add_argument('-m', type=str, required=True, help='Model name')

    opt = parser.parse_args()
    iters = int(5e5) + 1
    train(opt.m, opt.t, opt.v, iters)
