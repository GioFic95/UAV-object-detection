import argparse
import string
import numpy as np
import cv2

chars = string.digits + string.ascii_uppercase + string.ascii_lowercase


def prova(model_name, training_dataset, validation_dataset, train_steps=5e5, pretrained=None, fine_tuning=False,
          learning_rate=1e-3):
    print(model_name, training_dataset, validation_dataset, train_steps, pretrained, fine_tuning, learning_rate)
    print(type(model_name), type(training_dataset), type(validation_dataset), type(train_steps), type(pretrained),
          type(fine_tuning), type(learning_rate))


if __name__ == '__main__':
    print(len(chars), chars)
    c = 30
    print(c, chars[c-1])

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
    prova(opt.m, opt.t, opt.v)

    prova(opt.m, opt.t, opt.v, opt.i, opt.p, opt.f, opt.l)
