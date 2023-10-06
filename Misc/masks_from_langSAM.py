import os
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image

from util import segmentation_lang

warnings.filterwarnings("ignore", category=UserWarning)

# set parameters
image_dir = '/home/michael/Master/Datasets/Dataset_Flat_SAM/train/Images'
label_dir = '/home/michael/Master/Datasets/Dataset_Flat_SAM/train/Labels'

for file_name in tqdm(os.listdir(image_dir)):
    image_path = os.path.join(image_dir, file_name)
    jpg_image = Image.open(image_path)
    mask_array = segmentation_lang(jpg_image)
    png_array = mask_array.astype(np.uint8)
    mask_image = Image.fromarray(png_array)
    mask_path = f'{os.path.join(label_dir, file_name[:-4])}.png'
    mask_image.save(mask_path)