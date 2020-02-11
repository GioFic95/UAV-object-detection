import random
import glob
import os
from shutil import copyfile


src = "out_img"
dst = "sample"

imgs = glob.glob('./out_img/*.png')
for i in range(1000):
    rand_index = random.randint(0, len(imgs))
    src_file = imgs[rand_index]
    name = os.path.basename(src_file).split('_')
    print(name)
    dst_file = dst + '/' + name[0] + '_' + name[1] + '_' + str(i) + ".png"
    copyfile(src_file, dst_file)
