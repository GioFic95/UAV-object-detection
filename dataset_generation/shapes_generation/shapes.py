from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import string
import random
import os
import glob
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imageio

"""per ogni coppia forma-carattere:
  	1. peschiamo __K__ immagini random dalla cartella img_in, e da ciascuno un settore random della dimensione desiderata (244x244), da usare come sfondo
  	2. per ognuno di questi:
				3. Generiamo __N__ immagini in cui la forma varia per dimensione tra 150 e 170 e il carattere cambia dimensione e font
				4. Per ogni immagine generata:
						5. usiamo ImageDataGenerator e OpenCV per ottenere __M__ immagini modificate random
                            (rotazione, shift, luminosità, shear, zoom, channel shift, contrasto, sfocatura)
                            
total number of pictures = #shapes * #chars * #fonts * M"""


input_images = glob.glob("./in_img/*.jpg")
shapes = glob.glob("./shapes/*.png")
out_path = "./out_img"
print("\n\n", input_images, "\n\n", shapes, "\n\n", out_path)

chars = "ABCDEFGHIJKLMNIOPQRTUVWXYZ0123456789"

colors = [(0,0,0,255),\
          (255,255,255,255),\
          (150,0,0,255),\
          (0,150,0,255),\
          (0,0,150,255),\
          (128,128,128,255),\
          (255,234,0,255),\
          (128,0,128,255),\
          (165,104,42,255),\
          (255,165,0,255)]

size = 244
box_size = 150
font_size = int(box_size/2)

num_shapes = len(shapes)
num_chars = len(chars)
num_fonts = len(fonts)

M = 10

MAX_BLUR = 4
MIN_CONTRAST = 0.3



def get_rand_pos(base):
    x = random.randint(0, base.width - size)
    y = random.randint(0, base.height - size)
    return x, y


def phase1(fonts, fonts_path):
    with open("log.csv", "w") as f:
        f.write("char,shape,base,font,w,h,blur,contrast\n")
    i = 1
    
    for s in shapes:
        white_shape = Image.open(s).convert('RGBA')
        width = white_shape.width
        height = white_shape.height
        w = box_size if width > height else int((width * box_size)/height)
        h = box_size if width < height else int((height * box_size)/width)
        white_shape = white_shape.resize((w, h))

        for char in chars:     
            for font in fonts:
                data = np.array(white_shape)   # "data" is a height x width x 4 numpy array
                red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

                # Replace white with random color
                white_areas = (red == 255) & (blue == 255) & (green == 255)
                random_color = random.choice(colors)
                data[..., :-1][white_areas.T] = random_color[:3] # Transpose back needed
                shape = Image.fromarray(data)

                # get a background image
                img = random.choice(input_images)
                base = Image.open(img).convert('RGBA')

                x, y = get_rand_pos(base)
                base = base.crop((x, y, x+size, y+size))                

                # get a font
                fnt = ImageFont.truetype(fonts_path + font, font_size)
                # make a blank image for the text, initialized to transparent text color
                txt = Image.new('RGBA', (size, size), (255,255,255,0))
                # get a drawing context
                d1 = ImageDraw.Draw(txt)
                # draw text
                center = int(size/2)
                char_size = fnt.getsize(char)
                xy1 = center - int(char_size[0]/2), center - int(char_size[1]/2)

                char_color = random.choice(colors)
                while (char_color == random_color):
                    char_color = random.choice(colors)
                d1.text(xy1, char, font=fnt, fill=char_color)

                # put text, shape and base together
                xy2 = center - int(w/2), center - int(h/2)
                base.alpha_composite(shape, xy2)
                out = Image.alpha_composite(base, txt)
                
                # apply blur and change contrast
                blur = random.randint(1, MAX_BLUR)
                out = out.filter(ImageFilter.GaussianBlur(blur))
                contrast = random.uniform(MIN_CONTRAST, 1)
                out = ImageEnhance.Contrast(out).enhance(contrast)
                
                # save result image and append info about this image
                base_name, _ = os.path.splitext(os.path.basename(img))
                base_name = base_name.replace('(', '').replace(')', '').replace(' ', '_')
                shape_name, _ = os.path.splitext(os.path.basename(s))
                shape_name = shape_name.split("_")[0]
                font_name, _ = os.path.splitext(font)
                salt = random.randint(1000000, 9999999)
                
                file_name = shape_name + "_" + char + "_" + str(salt) + ".png"
                img_out = os.path.join(out_path, file_name)
                out.save(img_out, "PNG")
                
                with open("log.csv", "a") as f:
                    f.write(char + "," + shape_name + "," + base_name + "," + font_name + "," + str(w) + "," + str(h) + "," + str(blur) + "," + str(contrast) + "\n")
                    
                # print actual state
                print(file_name + "   -   " + str(100*(i/(num_shapes*num_chars*num_fonts))) + "%")
                i += 1


# def phase2():
#     datagen = ImageDataGenerator(rotation_range=180, width_shift_range=0.2, 
#     height_shift_range=0.2,shear_range=0, 
#     zoom_range=0,channel_shift_range = 0, horizontal_flip=False)

#     intermediate_images = glob.glob(out_path + "/*.png")
#     for image_path in intermediate_images:
#         image = np.expand_dims(imageio.imread(image_path), 0)

#         save_here = './out_img'

#         datagen.fit(image)

#         flow = datagen.flow(image,         # image we chose
#                 save_to_dir=save_here,     # this is where we figure out where to save
#                 save_prefix='aug',         # it will save the images as 'aug_0912' some number for every new augmented image
#                 save_format='png')
            
#         for x in range(10):
#             flow.next()


if __name__ == "__main__":
    fonts_path_gab = "/usr/share/fonts/truetype/dejavu/"
    fonts_gab = ["DejaVuMathTeXGyre.ttf"]
    fonts_path_gio = "C:\Windows\Fonts\\"
    fonts_gio = ["Arial.ttf", "Roboto-regular.ttf", "Times.ttf", "Cambria.ttc", "Comic.ttf", "Verdana.ttf", "Pala.ttf", "Georgia.ttf", "Calibri.ttf", "Javatext.ttf"]
    phase1(fonts_gio, fonts_path_gio)
    # phase1(fonts_gab, fonts_path_gab)
    # phase2()

