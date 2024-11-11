import os

import argparse
import glob
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from elias.util import save_img

from eg3d_preprocessor.preprocess.extract_camera import CameraExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input_root', type=str, default='./test/images/')
    parser.add_argument('--output_root', type=str, default='./test/dataset/')
    parser.add_argument('--mode', type=str, default='jpg')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Real dataset
    input_dir = args.input_root
    root = args.output_root

    c_out = os.path.join(root, f'c')
    crop_out = os.path.join(root, f'crop')

    os.makedirs(c_out, exist_ok=True)
    os.makedirs(crop_out, exist_ok=True)

    extractor = CameraExtractor()

    mode = args.mode
    image_list = sorted(glob.glob(f'{input_dir}/*.{mode}'))

    for image_path in tqdm.tqdm(image_list):
        image_name = Path(image_path).stem
        image = np.asarray(Image.open(image_path).convert('RGB'))
        camera, image_cropped = extractor.extract(image)

        np.save(f"{c_out}/{image_name}.npy", camera)
        save_img(image_cropped, f"{crop_out}/{image_name}.{mode}")


if __name__ == '__main__':
    main()