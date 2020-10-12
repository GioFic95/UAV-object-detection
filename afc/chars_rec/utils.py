from pprint import pprint
import os
import shutil
import glob
import argparse
import string
import numpy as np
import cv2
import torch
from torch.utils import data
import torchvision
import torchvision.models as models
import tensorflow as tf
import matplotlib.pyplot as plt

chars = string.digits + string.ascii_uppercase + string.ascii_lowercase


def prova(model_name, training_dataset, validation_dataset, train_steps=5e5, pretrained=None, fine_tuning=False,
          learning_rate=1e-3):
    print(model_name, training_dataset, validation_dataset, train_steps, pretrained, fine_tuning, learning_rate)
    print(type(model_name), type(training_dataset), type(validation_dataset), type(train_steps), type(pretrained),
          type(fine_tuning), type(learning_rate))


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        plt.savefig("grid.png")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("grid.png")


def split_dataset():
    traindir = "../datasets/English/Img/GoodImg/Bmp"

    train_loader = data.DataLoader(
        torchvision.datasets.ImageFolder(traindir, torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
        batch_size=10, shuffle=True,
        num_workers=6, pin_memory=True)

    print(len(train_loader))

    for i, (images, labels) in enumerate(train_loader):
        print(labels)
        classes = [chars[x] for x in labels]
        print(classes)
        img_grid = torchvision.utils.make_grid(images, nrow=5)
        matplotlib_imshow(img_grid, one_channel=False)

        if i == 0:
            break


def copy_datasets():
    os.chdir("./chars74k-master")
    train_path = "./datasplits/good_train"
    val_path = "./datasplits/good_validation"
    test_path = "./datasplits/good_test"
    new_train = "../../datasets/mychars/train"
    new_val = "../../datasets/mychars/val"
    new_test = "../../datasets/mychars/test"
    old_paths = [train_path, val_path, test_path]
    new_paths = [new_train, new_val, new_test]

    for old, new in zip(old_paths, new_paths):
        with open(old) as f:
            files = f.read().splitlines()
            for img in files:
                name = os.path.basename(img)
                new_dir = img.split("/")[-2]
                new_path = os.path.join(new, new_dir)
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                new_name = os.path.join(new_path, name)
                shutil.copy2(img, new_name)


if __name__ == '__main__':
    # print(len(chars), chars)
    # c = 30
    # print(c, chars[c-1])
    #
    # print(tf, torch)
    #
    # split_dataset()
    #
    # model_names = sorted(name for name in models.__dict__
    #                      if name.islower() and not name.startswith("__")
    #                      and callable(models.__dict__[name]))
    # print(model_names)

    copy_datasets()
