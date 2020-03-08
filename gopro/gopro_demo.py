import datetime
import os
import itertools
import pandas as pd
from goprocam import GoProCamera
from goprocam import constants
from collections import OrderedDict

TIMER = 1
settings = OrderedDict({
    constants.Photo.RAW_PHOTO: [
        constants.Photo.RawPhoto.ON,        # Raw photo
        constants.Photo.RawPhoto.OFF],      # JPEG photo
    constants.Photo.SUPER_PHOTO: [
        constants.Photo.SuperPhoto.Auto,
        constants.Photo.SuperPhoto.HDROnly,
        constants.Photo.SuperPhoto.OFF],
    constants.Photo.PROTUNE_PHOTO: [constants.Photo.ProTune.OFF, constants.Photo.ProTune.ON],
    constants.Photo.WHITE_BALANCE: [
        constants.Photo.WhiteBalance.WBAuto,
        constants.Photo.WhiteBalance.WBNative,
        constants.Photo.WhiteBalance.WB3000k,
        constants.Photo.WhiteBalance.WB5500k,
        constants.Photo.WhiteBalance.WB6500k],
    constants.Photo.COLOR: [constants.Photo.Color.Flat, constants.Photo.Color.GOPRO],
    constants.Photo.SHARPNESS: [constants.Photo.Sharpness.High],
    constants.Photo.RESOLUTION: [
        constants.Photo.Resolution.R12W,  # Resolution wide
        constants.Photo.Resolution.R12L],  # Resolution linear
})

tests = pd.DataFrame(list(itertools.product(*list(settings.values()))), columns=list(settings.keys()))
print(tests.columns, tests.shape)
gpCam = GoProCamera.GoPro()
print(gpCam.getStatusRaw())


def print_status_summary():
    print(gpCam.getStatus("settings", constants.Photo.RAW_PHOTO),
          gpCam.getStatus("settings", constants.Photo.SUPER_PHOTO),
          gpCam.getStatus("settings", constants.Photo.PROTUNE_PHOTO),
          gpCam.getStatus("settings", constants.Photo.WHITE_BALANCE),
          gpCam.getStatus("settings", constants.Photo.COLOR),
          gpCam.getStatus("settings", constants.Photo.SHARPNESS),
          gpCam.getStatus("settings", constants.Photo.RESOLUTION))


def take_pic(name):
    new_pic = gpCam.take_photo(TIMER)
    print("new_pic:", new_pic)
    fn = os.path.splitext(gpCam.getInfoFromURL(new_pic)[1])
    fn = fn[0] + "__" + name + fn[1]
    print("fn:", fn)
    gpCam.downloadLastMedia(new_pic, "pics/demo/" + fn)


print_status_summary()
print("starting:", datetime.datetime.now())
for i, row in tests.iterrows():
    name = "_".join(row.values) + '_0'
    for col, elem in row.iteritems():
        gpCam.gpControlSet(col, elem)
    print_status_summary()
    take_pic(name)
print("adding zoom:", datetime.datetime.now())
gpCam.setZoom(100)
for i, row in tests.iterrows():
    name = "_".join(row.values) + '_1'
    for col, elem in row.iteritems():
        gpCam.gpControlSet(col, elem)
    take_pic(name)
print("finishing:", datetime.datetime.now())

# gpCam.gpControlSet(constants.Photo.RESOLUTION, constants.Photo.Resolution.R12W)
# take_pic("res_wide")
# gpCam.gpControlSet(constants.Photo.RESOLUTION, constants.Photo.Resolution.R12L)
# take_pic("res_linear")
