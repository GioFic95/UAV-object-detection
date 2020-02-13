from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


shapes = glob.glob("./shapes/*.png")
chars = "ABCDEFGHIJKLMNIOPQRTUVWXYZ0123456789"

white = (255, 255, 255, 255)
black = (0, 0, 0, 255)

background_color = black 
char_color = black


size = 244
box_size = 150
font_size = int(box_size/2)

# numero di immagini prodotte dal Keras Generator
N = 10


def generate_bw_iamges(fonts, fonts_path, out_dir):
    num_shapes = len(shapes)
    num_chars = len(chars)
    num_fonts = len(fonts)
    num_images = num_shapes*num_chars*num_fonts
    num_augmented_imgs = N*num_images
    
    datagen = ImageDataGenerator(rotation_range=360,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=10,
                                 zoom_range=0,
                                 brightness_range=(0.8, 1),
                                 channel_shift_range=0,
                                 horizontal_flip=False,
                                 fill_mode='wrap')

    with open("log_bw.csv", "w") as f:
        f.write("file_name, char, shape\n")
    image_id = 0
    
    for s in shapes:
        white_shape = Image.open(s).convert('RGBA')
        
        width = white_shape.width
        height = white_shape.height
        w = box_size if width > height else int((width * box_size)/height)
        h = box_size if width < height else int((height * box_size)/width)
        white_shape = white_shape.resize((w, h))
        shape_name, _ = os.path.splitext(os.path.basename(s))
        shape_name = shape_name.split("_")[0]

        for char in chars:     
            for font in fonts:
                base = Image.new('RGBA', (size, size), background_color)               

                # get a font
                fnt = ImageFont.truetype(fonts_path + font, font_size)
                # make a blank image for the text, initialized to transparent text color
                txt = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                # get a drawing context
                d1 = ImageDraw.Draw(txt)
                # draw text
                center = int(size/2)
                char_size = fnt.getsize(char)
                xy1 = center - int(char_size[0]/2), center - int(char_size[1]/2)
                d1.text(xy1, char, font=fnt, fill=char_color)

                # put text, shape and base together
                xy2 = center - int(w/2), center - int(h/2)
                base.alpha_composite(white_shape, xy2)
                out = Image.alpha_composite(base, txt)
                font_name, _ = os.path.splitext(font)
                
                image_id += 1
                file_name = shape_name + "_" + char + "_" + str(image_id)
                img_out = os.path.join(out_dir, file_name + ".png")
                out.save(img_out, "PNG")
                
                with open("log_bw.csv", "a") as f:
                    f.write(file_name + ".png" + "," + char + "," + shape_name + "\n")
                    
                # print actual state
                print(file_name + ".png" + " - " + str(100*(image_id/(num_images + num_augmented_imgs))) + "%")
                
                image = np.expand_dims(out, 0)
                datagen.fit(image)
                flow = datagen.flow(image,  # image we chose
                                    save_to_dir=out_dir,  # this is where we figure out where to save
                                    save_prefix=file_name + "_aug",  # it will save the images with file_name prefix
                                    save_format='png')
                
                for x in range(N):
                    flow.next()
                    with open("log_bw.csv", "a") as f:
                        f.write(file_name + "aug" + "," + char + "," + shape_name + "\n")
                    image_id += 1


def main():
    fonts_path_gab = "/usr/share/fonts/truetype/dejavu/"
    fonts_gab = ["DejaVuMathTeXGyre.ttf"]

    fonts_path_gio = "C:\Windows\Fonts\\"
    fonts_gio = ["Arialbd.ttf", "Roboto-bold.ttf", "Times.ttf", "Cambria.ttc", "Verdana.ttf"]
    fonts_gio2 = ["Arialbd.ttf"]

    fonts_path_nene = "/Library/Fonts/"
    fonts_nene = ["Arial.ttf", "Andale Mono.ttf", "Arial Bold.ttf", "Verdana Bold.ttf"]

    out_dir = "./out_bw_img"
    generate_bw_iamges(fonts_gio2, fonts_path_gio, out_dir)


if __name__ == "__main__":
    main()
