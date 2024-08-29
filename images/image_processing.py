import os 
import argparse
from glob import glob

from tqdm import tqdm 
from PIL import Image

from preprocessing_utils import suppress_rotation

parser = argparse.ArgumentParser()
parser.add_argument("img_dir", help="path to the directory containing images to process")
parser.add_argument("save_path", help="path to the directory where to save the processed images")


def preprocess_images(dir , save_path):

    """ Utils function to apply preprocessing to the images and 
    save the result in a specified directory
    ------------------------------------------
    Args :
        dir : path to source directory where the images to process are located
        save_path : path to the destination directory
    """

    img_paths = glob(os.path.join(args.img_dir, '*.jpg'))

    for path in tqdm(img_paths):
        img_id = os.path.basename(path)
        image = Image.open(path)
        processed_image = suppress_rotation(image)
        processed_image.save(os.path.join(save_path,img_id))
        


if __name__ == '__main__':
    args = parser.parse_args()
    print("=" * 10, "Beginning of the processing", "=" * 10)
    preprocess_images(args.img_dir , args.save_path)
    print("=" * 10, "Processing Complete", "=" * 10)

