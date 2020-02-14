import random
import glob
import os
from shutil import copyfile


def pick_sample(src, dst, n):
    imgs = glob.glob(src + '/*.png')
    for i in range(n):
        rand_index = random.randint(0, len(imgs)-1)
        src_file = imgs[rand_index]
        name = os.path.basename(src_file).split('_')
        print(name)
        dst_file = dst + '/' + name[0] + '_' + name[1] + '_' + str(i) + ".png"
        copyfile(src_file, dst_file)


if __name__ == '__main__':
    pick_sample("./out_img", "sample", 8000)
