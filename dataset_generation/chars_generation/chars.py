from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


chars = "ABCDEFGHIJKLMNIOPQRTUVWXYZ0123456789"

colors = [(0, 0, 0, 255),
          (255, 255, 255, 255),
          (150, 0, 0, 255),
          (0, 150, 0, 255),
          (0, 0, 150, 255),
          (128, 128, 128, 255),
          (255, 234, 0, 255),
          (128, 0, 128, 255),
          (165, 104, 42, 255),
          (255, 165, 0, 255)]

size = 150
font_size = int(size/2)

# numero di immagini prodotte dal Keras Generator
N = 10
# numero di immagini prodotte modificando Blur e Contrasto
M = 10

MAX_BLUR = 3
MIN_CONTRAST = 0.5
MIN_SATURATION = 0.5
MAX_SATURATION = 1.5


def get_rand_pos(base):
    x = random.randint(0, base.width - size)
    y = random.randint(0, base.height - size)
    return x, y


def phase1(fonts, fonts_path, out_dir):
    num_chars = len(chars)
    num_fonts = len(fonts)
    num_images = num_chars*num_fonts
    num_augmented_imgs = N*num_images
    print("tot images:", num_augmented_imgs + num_images)
    
    datagen = ImageDataGenerator(rotation_range=360,
                                 width_shift_range=0.1, 
                                 height_shift_range=0.1,
                                 shear_range=10,
                                 zoom_range=0,
                                 brightness_range=(0.8, 1),
                                 channel_shift_range=0,
                                 horizontal_flip=False,
                                 fill_mode='wrap')

    with open("log.csv", "w") as f:
        f.write("file_name, char\n")
    image_id = 0
    
    for char in chars:     
        for font in fonts:
            # get a font
            fnt = ImageFont.truetype(fonts_path + font, font_size)
            # make a blank image for the text, initialized to transparent text color
            bg_color = random.choice(colors)
            txt = Image.new('RGBA', (size, size), bg_color)
            # get a drawing context
            d1 = ImageDraw.Draw(txt)
            # draw text
            center = int(size/2)
            char_size = fnt.getsize(char)
            xy1 = center - int(char_size[0]/2), center - int(char_size[1]/2)

            char_color = random.choice(colors)
            while char_color == bg_color:
                char_color = random.choice(colors)
            d1.text(xy1, char, font=fnt, fill=char_color)

            # put text, shape and base together
#            out = Image.alpha_composite(base, txt)
            out = txt
            
            # save result image and append info about this image
            image_id += 1
            file_name = char + "_" + str(image_id)
            img_out = os.path.join(out_dir, file_name + ".png")
            out.save(img_out, "PNG")
            
            with open("log.csv", "a") as f:
                f.write(file_name + ".png" + "," + char + "\n")
                
            # print actual state
            print(file_name + ".png" + " - " + str(100*(image_id/(num_images + num_augmented_imgs))) + "%")
            
            image = np.expand_dims(out, 0)
            # datagen.fit(image)
            flow = datagen.flow(image,  # image we chose
                                save_to_dir=out_dir,  # this is where we figure out where to save
                                save_prefix=file_name + "_aug",  # it will save the images with file_name prefix
                                save_format='png')
            
            for x in range(N):
                flow.next()
                with open("log.csv", "a") as f:
                    f.write(file_name + "aug" + "," + char + "\n")
                image_id += 1


def phase2(int_dir, out_dir):
    j = 0   # number of created images
    images = glob.glob(int_dir + "/*.png")

    for image_path in images:
        image = Image.open(image_path).convert('RGBA')
        
        name, ext = os.path.splitext(os.path.basename(image_path))
            
        char = name.split("_")[0]

        for i in range(M):
            # apply blur and change contrast and saturation
            blur = random.randint(1, MAX_BLUR)
            out = image.filter(ImageFilter.GaussianBlur(blur))
            contrast = random.uniform(MIN_CONTRAST, 1)
            out = ImageEnhance.Contrast(out).enhance(contrast)
            # saturation = random.uniform(MIN_SATURATION, MAX_SATURATION)
            # out = ImageEnhance.Color(out).enhance(saturation)
            
            j += 1
            new_name = name + "_" + str(j) + ext
            out.save(os.path.join(out_dir, new_name), "PNG")
            
            with open("log.csv", "a") as f:
                f.write(new_name + "," + char + "\n")
            
            print(new_name + " --- " + str(100*(j/(len(images)*M))))


def main():
    fonts_path_gab = "/usr/share/fonts/truetype/dejavu/"
    fonts_gab = ["DejaVuMathTeXGyre.ttf"]

    fonts_path_gio = "C:\Windows\Fonts\\"
    fonts_gio = ["Arialbd.ttf", "Roboto-bold.ttf", "Times.ttf", "Cambria.ttc", "Verdana.ttf"]

    fonts_path_nene = "/Library/Fonts/"
    fonts_nene = ["Arial.ttf", "Andale Mono.ttf", "Arial Bold.ttf", "Verdana Bold.ttf"]

    intermediate_path = "./int_img"
    out_path = "./out_img"
    

    phase1(fonts_gio, fonts_path_gio, intermediate_path)
    phase2(intermediate_path, out_path)


if __name__ == "__main__":
    main()
