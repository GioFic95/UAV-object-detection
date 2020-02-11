from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import string
import random
import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imageio



input_images = glob.glob("./in_img/*.jpg")
shapes = glob.glob("./shapes/*.png")

out_path = "./out_img"
print("\n\n", input_images, "\n\n", shapes, "\n\n", out_path)

#chars = "ABCDEFGHIJKLMNIOPQRTUVWXYZ0123456789"
chars = "A"

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

N = 10 # numero di immagini prodotte dal Keras Generator 
M = 1 # numero di immagini prodotte modificando Blur e Contrasto

MAX_BLUR = 3
MIN_CONTRAST = 0.5
MIN_SATURATION = 0.5
MAX_SATURATION = 1.5


def get_rand_pos(base):
    x = random.randint(0, base.width - size)
    y = random.randint(0, base.height - size)
    return x, y


def phase1(fonts, fonts_path):
    num_shapes = len(shapes)
    num_chars = len(chars)
    num_fonts = len(fonts)
    
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
                data = np.array(white_shape)   # "data" is a height x width x 4 numpy array
                red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

                # Replace white with random color
                white_areas = (red == 255) & (blue == 255) & (green == 255)
                shape_color = random.choice(colors)
                data[..., :-1][white_areas.T] = shape_color[:3] # Transpose back needed
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
                while (char_color == shape_color):
                    char_color = random.choice(colors)
                d1.text(xy1, char, font=fnt, fill=char_color)

                # put text, shape and base together
                xy2 = center - int(w/2), center - int(h/2)
                base.alpha_composite(shape, xy2)
                out = Image.alpha_composite(base, txt)
                
                # save result image and append info about this image
                base_name, _ = os.path.splitext(os.path.basename(img))
                base_name = base_name.replace('(', '').replace(')', '').replace(' ', '_')
                font_name, _ = os.path.splitext(font)
                
                image_id += 1
                file_name = shape_name + "_" + char + "_" + str(image_id) + ".png"
                img_out = os.path.join(out_path, file_name)
                out.save(img_out, "PNG")
                
                with open("log.csv", "a") as f:
                    f.write(file_name + "," + char + "," + shape_name + "\n")
                    
                # print actual state
                print(file_name + " - " + str(100*(image_id/(N*num_shapes*num_chars*num_fonts))) + "%")
                
                image = np.expand_dims(out, 0)
                datagen.fit(image)
                
                for x in range(N):
                    flow = datagen.flow(image,     # image we chose
                        save_to_dir=out_dir,       # this is where we figure out where to save
                        save_prefix=file_name + "aug",  # it will save the images with file_name prefix
                        save_format='png')
                    flow.next()
                    with open("log.csv", "a") as f:
                        f.write(file_name + "aug" + "," + char + "," + shape_name + "\n")
                    image_id += 1





def phase2(intermediate_images, out_dir):
    j = 0 # number of created images
    for image_path in intermediate_images:
        image = Image.open(image_path).convert('RGBA')
        
        name, ext = os.path.splitext(os.path.basename(image_path))
            
        shape_name = name.split("_")[0]
        char = name.split("_")[1]

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
            out.save(new_name, "PNG")
            
            with open("log.csv", "a") as f:
                f.write(new_name + "," + char + "," + shape_name + "\n")
            
            print(os.path.join(out_dir, new_name) + " --- " + str(100*(j/(len(intermediate_images)*M))))




if __name__ == "__main__":
    fonts_path_gab = "/usr/share/fonts/truetype/dejavu/"
    fonts_gab = ["DejaVuMathTeXGyre.ttf"]
    
    fonts_path_gio = "C:\Windows\Fonts\\"
    fonts_gio = ["Arialbd.ttf", "Roboto-bold.ttf", "Times.ttf", "Cambria.ttc", "Comic.ttf", "Verdana.ttf", "Framd.ttf", "Georgia.ttf", "Calibri.ttf", "Javatext.ttf"]
    
    fonts_path_nene = "/Library/Fonts/"
    fonts_nene = ["Arial.ttf", "Andale Mono.ttf", "Arial Bold.ttf", "Verdana Bold.ttf"]
    
    # fonts_gio = ["Arial.ttf"]
    intermediate_images = glob.glob(out_path + "/*.png")
    out_dir = './out_img'

    phase1(fonts_gab, fonts_path_gab)
    phase2(intermediate_images, out_dir)

