import os
import argparse
from glob import glob

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel

from utils import  process_image  


# Setting the device
device = 'cuda:0' if torch.cuda.is_available() else "cpu"

processor = DonutProcessor.from_pretrained("Edgar404/donut-plate-recognition-best_model")
model = VisionEncoderDecoderModel.from_pretrained("Edgar404/donut-plate-recognition-best_model")
model.to(device)

parser = argparse.ArgumentParser()
parser.add_argument("img_dir", help="The path to the images to predict")
parser.add_argument("--save_path", help="Path to the CSV file where the predictions \
                                          will be saved. If the file doesn't exist, it will be created.")


def predict(path):
    processed_image = Image.open(path)
    output = process_image(processed_image, model, processor, d_type=torch.float32)
    return output['plate_number']


def make_prediction(args):
    img_paths = glob(os.path.join(args.img_dir, '*.jpg'))
    rows = []

    for path in tqdm(img_paths):
        img_id = os.path.basename(path)
        plate_number = predict(path)
        rows.append({"img_id": img_id, "plate_number": plate_number})

    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    args = parser.parse_args()
    print("=" * 10, "Beginning parsing", "=" * 10)
    df = make_prediction(args)
    print("=" * 10, "Parsing Complete", "=" * 10)

    print("=" * 10, "Saving the file", "=" * 10)
    if args.save_path is not None:
        df.to_csv(args.save_path)
    else:
        df.to_csv('predictions.csv' , index=False)
    print("=" * 10, "Save Complete", "=" * 10)
