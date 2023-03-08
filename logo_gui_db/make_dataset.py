import cairosvg
from PIL import Image
import numpy as np

def svg2png(load, save, h=128, w=128):
    cairosvg.svg2png(url=load, write_to=save, output_height=h, output_width=w)

def png_bk_color(load, save):
    f = Image.open(load) 
    data = np.array(f)
    a = data[:,:,3]
    # find opaque pixels
    mask = (a==0)
    # set alpha channel to 255
    data[:,:,3] = 255
    # plot opaque pixels by white pixels
    data[:,:,:3][mask] = [255,255,255]
    f_new = Image.fromarray(data)
    f_new = f_new.resize(f.size)
    f_new.save(save)

def watermark(mark_path, img_path, alpha):
    mark = Image.open(mark_path)
    img = Image.open(img_path)
    mark_name = mark_path.split('/')[-1].split('.')[0]
    img_name = img_path.split('/')[-1].split('.')[0]
    img = img.resize(mark.size).convert('RGBA')
    marked_image = Image.blend(img, mark, alpha=alpha)
    marked_image.save(f"{mark_name}_{img_name}_{alpha}.png")

def superpose(back_path, fore_path, save_to=None):

    background = Image.open(back_path)
    foreground = Image.open(fore_path)
    fore_name = fore_path.split('/')[-1].split('.')[0]
    back_name = back_path.split('/')[-1].split('.')[0]
    foreground = foreground.convert('RGBA')
    background = background.resize(foreground.size).convert('RGBA')
    background.paste(foreground, (0, 0), foreground)
    background = background.convert('RGB')
    if save_to != None:
        background.save(save_to)
    else:
        background.save(f"{fore_name}_{back_name}.jpg")

def embed(img1_path, img2_path, mask_path, save_to):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    mask = Image.open(mask_path).convert('L')
    img1_name = img1_path.split('/')[-1].split('.')[0]
    img2_name = img2_path.split('/')[-1].split('.')[0]
    img1 = img1.resize(mask.size).convert('RGB')
    img2 = img2.resize(mask.size).convert('RGB')
    img1.paste(img2, (0, 0), mask)
    img1 = img1.convert('RGB')
    if save_to != None: 
        img1.save(save_to)
    else:
        img1.save(f"{img2_name}_{img1_name}.jpg")

def stats(img_path):
    f = Image.open(img_path)
    data = np.array(f)
    print({'mode':f.mode,
            'format':f.format,
            'shape':data.shape,
            })
    return data