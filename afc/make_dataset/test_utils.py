import h5py
import traceback
import cv2
import os
import random
import shutil
from zipfile import ZipFile
import numpy as np
from PIL import Image
from imutils import rotate, rotate_bound
from scipy import ndimage

dataset = ""


def test_h5():
    fname = "./SynthText/data/dset.h5"
    db = h5py.File(fname, 'r')
    print("keys:", db.keys())
    imnames = sorted(db['image'].keys())
    print(len(imnames))
    for i in range(min(10, len(imnames))):
        imname = imnames[i]
        try:
            img = db['image'][imname][:]
            print("img type:", type(img))
            depth = db['depth'][imname][:].T
            depth0 = depth[:, :, 0]
            depth1 = depth[:, :, 1]
            seg = db['seg'][imname][:].astype('float32')
            area = db['seg'][imname].attrs['area']
            label = db['seg'][imname].attrs['label']

            print(f"imname {imname}\n"
                  f"img shape {img.shape}\n"
                  f"depth0 {depth0.shape}, {depth0}\n"
                  f"depth1 {depth1.shape}, {depth1}\n"
                  f"seg {seg.shape}{seg}\n"
                  f"area {area.shape}{area}\n"
                  f"label {label.shape}{label}\n")
        except:
            traceback.print_exc()
            print("ERROOOOOOOOR")


def print_attrs(name, obj):
    shape = ""
    try:
        shape = obj.shape
    except:
        pass
    print(name, type(obj), shape)
    if type(obj) == h5py.Dataset:
        print(obj[:])
        try:
            cv2.imwrite("img/" + f"{dataset}_{name}.jpg".replace("/", "-"), cv2.cvtColor(obj[:], cv2.COLOR_BGR2RGB))
        except cv2.error:
            # cv2.imwrite("img/" + f"{dataset}_{name}_0.jpg".replace("/", "-"),
            #             cv2.cvtColor(obj[:].T[:, :, 0], cv2.COLOR_BGR2RGB))
            cv2.imwrite("img/" + f"{dataset}_{name}_1.jpg".replace("/", "-"),
                        cv2.cvtColor(obj[:].T[:, :, 1], cv2.COLOR_BGR2RGB))
    for key, val in obj.attrs.items():
        print("    %s: %s %s" % (key, type(val), val.shape))
        print(val)
    print()


def describe(ds):
    print(ds)
    global dataset
    dataset = os.path.basename(ds)
    f = h5py.File(ds, 'r')
    f.visititems(print_attrs)
    print()


def join(bg_path, seg_db_path, depth_db_path, out_path):
    bg = os.listdir(bg_path)
    seg_db = h5py.File(seg_db_path, 'r')
    depth_db = h5py.File(depth_db_path, 'r')

    assert list(seg_db["mask"].keys()) == list(depth_db.keys()) and list(depth_db.keys()) == bg
    print("merging databases...")

    # create h5 database
    out_db = h5py.File(out_path, 'w')

    # add images
    out_db.create_group('/image')
    for img_name in bg:
        img = np.asarray(Image.open(os.path.join(bg_path, img_name)))
        out_db['image'].create_dataset(img_name, data=img)

    # add depths
    out_db.create_group('/depth')
    for depth_name in depth_db:
        depth = [np.zeros_like(depth_db[depth_name]), depth_db[depth_name]]
        out_db['depth'].create_dataset(depth_name, data=depth)

    # add segments
    out_db.create_group('/seg')
    for seg_name in seg_db["mask"]:
        seg = seg_db["mask"][seg_name]
        out_db['seg'].create_dataset(seg_name, data=seg[:])
        out_db['seg'][seg_name].attrs['area'] = seg.attrs['area']
        out_db['seg'][seg_name].attrs['label'] = seg.attrs['label']

    # clos databases
    depth_db.close()
    seg_db.close()
    out_db.close()

    print("done")


def new_join(bg_path, seg_db_path, depth_db_path, out_path, depth='all_white'):
    bg = os.listdir(bg_path)
    seg_db = h5py.File(seg_db_path, 'r')
    depth_db = h5py.File(depth_db_path, 'r')
    starts = [(10, 10), (200, 100), (280, 400)]
    shape = [150, 150]
    depth_path = "./img/depth"

    print("merging databases...")

    # create h5 database
    out_db = h5py.File(out_path, 'w')
    out_db.create_group('/image')
    out_db.create_group('/depth')
    out_db.create_group('/seg')

    for i in range(len(bg)):
        start = starts[i]

        # add images
        img_name = bg[i]
        img = np.asarray(Image.open(os.path.join(bg_path, img_name)))
        out_db['image'].create_dataset(img_name, data=img)
        print(img.shape)

        # add depths
        depth_name = list(depth_db.keys())[i]
        depth0 = np.zeros_like(depth_db[depth_name])
        if depth == "seg_white":
            depth1 = np.zeros_like(depth_db[depth_name])
            depth1[start[1]:start[1] + shape[1], start[0]:start[0] + shape[0]] = 255.0
        elif depth == "gradient":
            depth1 = np.asarray(Image.open(os.path.join(depth_path, f"depth{i+1}.jpg")).convert('L')).T
        elif depth == "all_white":
            depth1 = np.full(depth_db[depth_name].shape, 255.0, dtype=np.float)
        else:
            depth1 = np.zeros_like(depth_db[depth_name])
        out_db['depth'].create_dataset(depth_name, data=[depth0, depth1])

        # add segments
        seg_name = list(seg_db["mask"].keys())[i]
        seg = np.zeros(depth0.T.shape, dtype=np.int)
        seg[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1]] = 255
        out_db['seg'].create_dataset(seg_name, data=seg)
        out_db['seg'][seg_name].attrs['area'] = np.array([0, 1000])
        out_db['seg'][seg_name].attrs['label'] = np.array([0, 255])

    # clos databases
    depth_db.close()
    seg_db.close()
    out_db.close()


def test_bb(img_path='./img/bg_true/DSC03373.JPG', out_path='./img/bg_true/test.jpg', starts=None, rots=None):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 450))

    if starts is None:
        starts = [
            [(62, 140), (109, 140), (62, 187), (109, 187)],
            [(257, 180), (298, 180), (257, 229), (298, 229)],
            [(457, 270), (495, 270), (457, 294), (495, 294)]
        ]
    w, h = 30, 35

    for i, start in enumerate(starts):
        locw = start[2][1] - start[0][1]
        loch = start[1][0] - start[0][0]
        loc = (start[0][1] + round(locw / 2 - h / 2), start[0][0] + round(loch / 2 - w / 2))
        print(locw, loch, loc)

        if rots is not None:
            print("rotate", rots[i])
            img = rotate(img, angle=rots[i])

        print(start[0], start[3])
        cv2.rectangle(img, start[0], start[3], (255, 0, 0), 1)
        cv2.rectangle(img, (loc[1], loc[0]), (loc[1] + w, loc[0] + h), (0, 0, 255), 1)

    cv2.imwrite(out_path, img)


def test_yield():
    import pandas as pd
    import albumentations
    SIZE = (600, 450)
    transform = albumentations.Compose(
        [albumentations.Resize(*SIZE)],
        bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )
    df = pd.read_csv("labeling/results/results.tsv", sep='\t')
    groups = df.groupby(by=['name'])
    for g in groups:
        name = g[0]
        path = os.path.join("labeling/test_img", name)
        image = cv2.imread(path)

        row = g[1]
        starts = [eval(x) for x in row["boundingBox"]]
        rots = row["rotation"].values
        shapes = row["shape"].values
        transformed = transform(image=image, bboxes=starts, class_labels=shapes)

        yield {"name": name, "image": transformed["image"], "starts": transformed["bboxes"], "rots": rots, "shapes": shapes}


def test_test_yield():
    for i, d in enumerate(test_yield()):
        print(i, d)
        print()


def all_landscapes(src_path, dst_path):
    with os.scandir(src_path) as it:
        for i, entry in enumerate(it):
            in_path = os.path.join(src_path, entry.name)
            out_path = os.path.join(dst_path, entry.name)
            img = cv2.imread(in_path)
            if img.shape[0] > img.shape[1]:
                img = ndimage.rotate(img, -90)
                print(f"{i}.\timage {entry.name} with shape {img.shape} --> rotated")
            else:
                print(f"{i}.\timage {entry.name} with shape {img.shape}")
            if not cv2.imwrite(out_path, img):
                print("can't save image")
                break


def split_ds(src_path, dst_path, n=100):
    images = os.listdir(src_path)
    random.shuffle(images)
    num = 0
    cur_dir = ""
    for i, img in enumerate(images):
        if i % n == 0:
            num += 1
            cur_dir = os.path.join(dst_path, str(num))
            try:
                os.mkdir(cur_dir)
            except FileExistsError:
                shutil.rmtree(cur_dir)
                os.mkdir(cur_dir)
        src = os.path.join(src_path, img)
        dst = os.path.join(cur_dir, img)
        shutil.copy(src, dst)
        print(f"{i}. image {src} copied into {dst}")


def zip_dirs(src_path, dst_path):
    with os.scandir(src_path) as it1:
        for i, dir in enumerate(it1):
            print(f"{i}. zipping dir {dir.name}")
            dir_path = os.path.join(src_path, dir.name)
            zip_name = os.path.join(dst_path, f"{dir.name}.zip")
            with ZipFile(zip_name, 'w') as zipObj:
                with os.scandir(dir_path) as it2:
                    for j, file in enumerate(it2):
                        file_path = os.path.join(dir_path, file.name)
                        zipObj.write(file_path, file.name)
                        print(f"{i}.{j}. file {file.name} zipped")


if __name__ == '__main__':
    # test_h5()
    # describe("./SynthText/results/SynthText.h5")
    # describe("./SynthText/data/dset.h5")
    # describe("./SynthText/data/seg_uint16.h5")
    # describe("./SynthText/data/depth.h5")
    # join("./SynthText/prep_scripts/bg", "./SynthText/data/seg_uint16.h5", "./SynthText/data/depth.h5", "test_db.h5")
    # describe("./SynthText/data/test_db.h5")
    # describe("./SynthText/results/TestSynthText.h5")
    # new_join("./SynthText/prep_scripts/bg", "./SynthText/data/seg_uint16.h5", "./SynthText/data/depth.h5", "SynthText/data/new_test_db3.h5")
    # describe("SynthText/data/new_test_db3.h5")
    # new_join("./SynthText/prep_scripts/true_bg", "./SynthText/data/seg_uint16.h5", "./SynthText/data/depth.h5", "SynthText/data/new_test_db4.h5")
    # describe("SynthText/data/new_test_db4.h5")
    # describe("./SynthText/results/NewTestSynthText.h5")
    # test_bb()

    # (272, 188, 48, 57) --> (255.0, 176.25, 45.0, 53.4375)
    # test_bb(img_path="./labeling/test_img/FakeTestSynthText.h5_data-DSC03373.JPG_19.jpg",
    #         out_path="./labeling/test_img/test.jpg",
    #         starts=[[(255, 176), (300, 176), (255, 230), (300, 230)]],
    #         rots=[-26])

    # test_test_yield()
    # all_landscapes("D:/Pictures/drone/10201113", "D:/Pictures/drone/uav_photos")
    # split_ds("D:/Pictures/drone/uav_photos", "D:/Pictures/drone/uav_split")
    zip_dirs("D:/Pictures/drone/uav_split", "D:/Pictures/drone/uav_zip")
