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

import albumentations
import pandas as pd

from .synthgen import *


def get_data(data_path, dataset_in, size, images_in, log):
    """
    Generator that yields the relevant fields from the dataset.
    """
    db_path = osp.join(data_path, dataset_in)
    log_path = osp.join(data_path, log)
    if not osp.exists(db_path):
        print(f"the database {db_path} does not exist")
        sys.exit(-1)
    transform = albumentations.Compose(
        [albumentations.Resize(*size)],
        bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )
    df = pd.read_csv(db_path, sep='\t')
    groups = df.groupby(by=['name'])
    for g in groups:
        name = g[0]
        path = os.path.join(data_path, images_in, name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        row = g[1]
        try:
            starts = [eval(x) for x in row["boundingBox"]]
            rots = row["rotation"].values
            shapes = row["shape"].values
            colors = row["shapeColor"].values
            transformed = transform(image=image, bboxes=starts, class_labels=shapes)
            tr_starts = np.array(transformed["bboxes"]).round().astype(np.int)
            print("STARTS:", tr_starts)

            yield {"name": name, "image": transformed["image"], "starts": tr_starts, "rots": rots,
                   "shapes": shapes, "colors": colors}
        except TypeError:
            error = f"continue for TypeError with img {name} (empty image)\n"
            print(error)
            with open(log_path, 'a') as lp:
                lp.write(error)
        except ValueError:
            error = f"continue for ValueError with img {name}\n"
            print(error)
            with open(log_path, 'a') as lp:
                lp.write(error)


def add_res_to_db(df, d, res, data_path, images_out):
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

        path = os.path.join(data_path, images_out, name)
        obj = res[i]['img']
        cv2.imwrite(path, cv2.cvtColor(obj, cv2.COLOR_BGR2RGB))

    return df


def gen_synth_ds(data_path, dataset_in, dataset_out, images_in, images_out, size, instance_per_image,
                 secs_per_img, log, viz=False):
    df_out = pd.DataFrame(columns=["name", "shapes", "shapeColors", "alphanumerics",
                                   "alphanumericColors", "boundingBoxes", "rotations"])
    RV3 = RendererV3(data_path, max_time=secs_per_img)

    for d in get_data(data_path, dataset_in, size, images_in, log):
        imname = d['name']
        print(f"\n*** processing image {imname} ***\n")
        try:
            # get the image:
            img = d['image']
            starts = d['starts']
            print("STARTS 1:", starts)
            rots = d['rots']
            res = RV3.render_text(img, starts, rots, ninstance=instance_per_image, viz=viz)
            # print("RES:", len(res), res[0]['txt'])
            if len(res) > 0:
                # non-empty : successful in placing text:
                df_out = add_res_to_db(df_out, d, res, data_path, images_out)
            # visualize the output:
            if viz:
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    df_out.to_csv(osp.join(data_path, dataset_out), index=False, sep='\t')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    args = parser.parse_args()

    # path to the data-file, containing image, depth and segmentation:
    data_path = 'data'
    dataset_in = "datasets/results.tsv"
    dataset_out = "datasets/full_results.tsv"
    images_in = "imgs_in"
    images_out = "imgs_out"
    log = "log.txt"
    size = (450, 600)

    # Define some configuration variables:
    instance_per_image = 3  # no. of times to use the same image
    secs_per_img = None  # 5  # max time per image in seconds

    gen_synth_ds(data_path, dataset_in, dataset_out, images_in, images_out, size, instance_per_image,
                 secs_per_img, log, args.viz)
