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

import h5py
import numpy as np
import tarfile
import wget
from PIL import Image

from common import *
from synthgen import *

# Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1  # no. of times to use the same image
# SECS_PER_IMG = 5  # max time per image in seconds
SECS_PER_IMG = None

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH, 'datasets/new_test_db4.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
# OUT_FILE = 'results/SynthText.h5'
OUT_FILE = 'results/NewTestSynthText.h5'
SIZE = (600, 450)
# starts = [
#     [
#         [(62, 140), (109, 140), (62, 187), (109, 187)],
#         [(257, 180), (298, 180), (257, 229), (298, 229)],
#         [(457, 270), (495, 270), (457, 294), (495, 294)]
#     ], [
#         [(124, 43), (155, 43), (124, 71), (155, 71)],
#         [(280, 99), (308, 99), (280, 142), (308, 142)]
#     ], [
#         [(119, 220), (153, 220), (119, 258), (153, 258)],
#         [(341, 303), (376, 303), (341, 337), (376, 337)],
#         [(528, 374), (563, 374), (528, 404), (563, 404)]
#     ],
# ]  # todo use real locations
starts = [
    [
        [62, 140, 109, 187],
        [257, 180, 298, 229],
        [457, 270, 495, 294]
    ], [
        [124, 43, 155, 71],
        [280, 99, 308, 142]
    ], [
        [119, 220, 153, 258],
        [341, 303, 376, 337],
        [528, 374, 563, 404]
    ],
]
rots = [
    [36, 23.8, 168.9],
    [198.7, 345.7],
    [129.4, 25.2, 194.3]
]  # todo use real rotations


def get_data():
    """
    Download the image,depth and segmentation data:
    Returns, the h5 database.
    """
    if not osp.exists(DB_FNAME):
        print("db_fname does not exist")
        try:
            colorprint(Color.BLUE, '\tdownloading data (56 M) from: ' + DATA_URL, bold=True)
            print()
            sys.stdout.flush()
            out_fname = 'data.tar.gz'
            wget.download(DATA_URL, out=out_fname)
            tar = tarfile.open(out_fname)
            tar.extractall()
            tar.close()
            os.remove(out_fname)
            colorprint(Color.BLUE, '\n\tdata saved at:' + DB_FNAME, bold=True)
            sys.stdout.flush()
        except:
            print(colorize(Color.RED, 'Data not found and have problems downloading.', bold=True))
            sys.stdout.flush()
            sys.exit(-1)
    # open the h5 file and return:
    return h5py.File(DB_FNAME, 'r')


def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        db['data'].create_dataset(dname, data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        # db['data'][dname].attrs['txt'] = res[i]['txt']
        L = res[i]['txt']
        L = [n.encode("ascii", "ignore") for n in L]
        db['data'][dname].attrs['txt'] = L


def main(viz=False):
    # open databases:
    print(colorize(Color.BLUE, 'getting data..', bold=True))
    db = get_data()
    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))

    # get the names of the image files in the dataset:
    imnames = sorted(db['image'].keys())
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)
    for i in range(start_idx, end_idx):
        imname = imnames[i]
        print(f"\n*** processing image {imname} ***\n")
        try:
            # get the image:
            img = Image.fromarray(db['image'][imname][:])

            # re-size uniformly:
            img = np.array(img.resize(SIZE, Image.ANTIALIAS))

            print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
            res = RV3.render_text(img, starts[i], rots[i], ninstance=INSTANCE_PER_IMAGE, viz=viz)
            print("RES:", len(res), res[0]['txt'])
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname, res, out_db)
            # visualize the output:
            if viz:
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    db.close()
    out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
