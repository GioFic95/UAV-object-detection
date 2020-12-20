# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import os
import os.path as osp
import sys
import traceback

import albumentations
import h5py
import numpy as np
import pandas as pd
import tarfile
import wget
from PIL import Image

from common import *
from synthgen import *

# Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 3  # no. of times to use the same image
# SECS_PER_IMG = 5  # max time per image in seconds
SECS_PER_IMG = None

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DATASET_IN = "datasets/results.tsv"
DATASET_OUT = "datasets/full_results.tsv"
IMAGES_IN = "imgs_in"
IMAGES_OUT = "imgs_out"
SIZE = (450, 600)


def get_data():
    """
    Generator that yields the relevant fields from the dataset.
    """
    db_path = osp.join(DATA_PATH, DATASET_IN)
    if not osp.exists(db_path):
        print(f"the database {db_path} does not exist")
        sys.exit(-1)
    transform = albumentations.Compose(
        [albumentations.Resize(*SIZE)],
        bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )
    df = pd.read_csv(db_path, sep='\t')
    groups = df.groupby(by=['name'])
    for g in groups:
        name = g[0]
        path = os.path.join(DATA_PATH, IMAGES_IN, name)
        image = cv2.imread(path)

        row = g[1]
        starts = [eval(x) for x in row["boundingBox"]]
        rots = row["rotation"].values
        shapes = row["shape"].values
        colors = row["shapeColor"].values
        transformed = transform(image=image, bboxes=starts, class_labels=shapes)
        tr_starts = np.array(transformed["bboxes"]).round().astype(np.int)
        print("STARTS:", tr_starts)

        yield {"name": name, "image": transformed["image"], "starts": tr_starts, "rots": rots,
               "shapes": shapes, "colors": colors}


def add_res_to_db(df, d, res):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    for i in range(len(res)):
        name, ext = os.path.splitext(d['name'])
        name = f"{name}_{i}{ext}"
        starts = str([list(s) for s in d['starts']])
        rots = str(list(d['rots']))
        shapes = str(list(d['shapes']))
        shape_colors = str(list(d['colors']))
        chars = res[i]['txt']
        char_colors = res[i]['color']
        df = df.append({"name": name, "shapes": shapes, "shapeColors": shape_colors, "alphanumerics": chars,
                       "alphanumericColors": char_colors, "boundingBoxes": starts, "rotations": rots},
                       ignore_index=True)

        path = os.path.join(DATA_PATH, IMAGES_OUT, name)
        obj = res[i]['img']
        # cv2.imwrite(path, cv2.cvtColor(obj[:].T[:, :, 1], cv2.COLOR_BGR2RGB))
        cv2.imwrite(path, obj)

    return df


def main(viz=False):
    df_out = pd.DataFrame(columns=["name", "shapes", "shapeColors", "alphanumerics",
                                   "alphanumericColors", "boundingBoxes", "rotations"])
    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)

    for d in get_data():
        imname = d['name']
        print(f"\n*** processing image {imname} ***\n")
        try:
            # get the image:
            img = d['image']
            starts = d['starts']
            print("STARTS 1:", starts)
            rots = d['rots']
            res = RV3.render_text(img, starts, rots, ninstance=INSTANCE_PER_IMAGE, viz=viz)
            # print("RES:", len(res), res[0]['txt'])
            if len(res) > 0:
                # non-empty : successful in placing text:
                df_out = add_res_to_db(df_out, d, res)
            # visualize the output:
            if viz:
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    df_out.to_csv(osp.join(DATA_PATH, DATASET_OUT), index=False, sep='\t')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
