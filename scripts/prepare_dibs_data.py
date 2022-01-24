from glob import glob
import os
import random

import numpy as np
from tqdm import tqdm

from matplotlib.image import imread, imsave
from PIL import UnidentifiedImageError


def rename(s):
    # Lactobacillus.johnsonii_0007.tif
    # Veionella_0009.tif
    s = s.replace('.tif', '')
    name, num = s.split('_')
    if '.' in name:
        name, suffix = name.split('.')
    else:
        suffix = 'sp'  # species
    folder = f'{name}_{suffix}'.lower()
    return folder, int(num)


def process_image(fp, outdir, bucket):
    try:
        im = imread(fp)
    except UnidentifiedImageError:
        return False
    # print(type(img))
    
    name, num = rename(os.path.basename(fp))

    imarray = np.array(im)
    im_h, im_w = imarray.shape[:2]
    block_h, block_w = 224, 224
    
    i = 1
    for row in np.arange(im_h - block_h + 1, step=block_h):
        for col in np.arange(im_w - block_w + 1, step=block_w):
            im1 = imarray[row: row + block_h, col: col + block_w, :]
            
            path = f'{outdir}/{bucket}/{name}/{name}_{num}_{i}.png'
            try:
                imsave(path, im1)
            except FileNotFoundError:
                os.makedirs(f'{outdir}/{bucket}/{name}')

            i += 1
    # print('Done.')
    return True



fp_img = 'data/DIBAS/img/*.tif'
outdir = 'data'
pct = [0.7, 0.15, 0.15]


failed = []
for img in tqdm(glob(fp_img)):
    draw = random.uniform(0, 1)
    if draw < pct[-1]:
        bucket = 'test'
    elif pct[-1] < draw < pct[-1] + pct[1]:
        bucket = 'val'
    else:
        bucket = 'train'
    
    if not process_image(img, outdir, bucket):
        failed.append(img)
        print(img)
