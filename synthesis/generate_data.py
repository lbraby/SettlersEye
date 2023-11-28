#!/usr/bin/env python3
#
import cv2
import os
import numpy as np
from tqdm import tqdm
import math
from scipy import ndimage
import random
import json
import re
import argparse

labels = ['blue-city', 'blue-road', 'blue-settlement', 'orange-city', 'orange-road', 'orange-settlement', 'red-city', 'red-road', 'red-settlement', 'white-city', 'white-road', 'white-settlement']

def place_piece(img: np.array, piece: np.array, location: tuple, scaling_factor: float) -> (np.array, list):
    piece = piece.copy()
    piece = cv2.resize(piece, (int(piece.shape[1] * scaling_factor), int(piece.shape[0] * scaling_factor)))

    piece = ndimage.rotate(piece, 23)

    y1, y2 = location[1] - math.ceil(piece.shape[0] / 2), location[1] + math.floor(piece.shape[0] / 2)
    x1, x2 = location[0] - math.ceil(piece.shape[1] / 2), location[0] + math.floor(piece.shape[1] / 2)

    alpha_piece = piece[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_piece

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_piece * piece[:, :, c] +
                                alpha_img * img[y1:y2, x1:x2, c])
    #cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 2)

    bbox = {"x": location[0], "y": location[1], "width":(x2-x1), "height": (y2-y1)}

    return img, bbox

def get_label(filepath: str):
    base = os.path.basename(filepath)
    results = re.search(r'.*(orange|red|white|blue).*(road|settlement|city).*', base)

    label = f"{results.group(1)}-{results.group(2)}"

    return label

def generate_data(background: np.array, images: str, n_images: int, start: int = 0, output_base: str = ""):
    for i in tqdm(range(start, start + n_images)):
        syn_img = background.copy()
        n_pieces = random.randrange(10, 61)
        annotations = {"image": f"synth{i}.png",
                       "annotations": []}
        for piece_path in random.choices(images, k=n_pieces):
            piece = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
            x = random.randrange(piece.shape[1]//2, syn_img.shape[1] - piece.shape[1]//2)
            y = random.randrange(piece.shape[0]//2, syn_img.shape[0] - piece.shape[0]//2)

            syn_img, bbox = place_piece(syn_img, piece, (x, y), .25)

            annotation = {"label": get_label(piece_path),
                          "coordinates": bbox
                          }

            annotations["annotations"].append(annotation)

        cv2.imwrite(os.path.join(output_base, f"../synthetic_images/synth{i}.png"), syn_img)
        with open(os.path.join(output_base, f"../synthetic_images/synth{i}.json"), "w") as json_file:
            json.dump([annotations], json_file)


def build_parser():
    parser = argparse.ArgumentParser(description="Python utility to generate synthetic data.")
    parser.add_argument("--background", "-b", metavar="FILE", help="Image data will be created from", required=False)
    parser.add_argument("--dir", '--directory', '-d', metavar="DIRECTORY", help="directory holding masks to generate from", required=False)
    parser.add_argument("--out", '--output', '-o', metavar="OUTPUT DIR", help="directory to put output files", required=False)
    parser.add_argument("--start", "-s", metavar="START", help="where to start numbering output files", required=False, default=0, type=int)
    parser.add_argument("--num", "-n", metavar="NUMBER", help="number of output files", required=False, default=10, type=int)
    # parser.add_argument("--crop", '-c', help="crop output files", required=False, action="store_true")

    return parser




def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.background:
        bg_path = args.background
    else:
        bg_path = os.path.join(os.path.dirname(__file__), "../synthetic_seed/boards/board1.jpg")

    syn_img = cv2.imread(bg_path)

    if args.dir:
        obj_path = args.dir
    else:
        obj_path = os.path.join(os.path.dirname(__file__), "../output")
    images = [os.path.join(obj_path, img) for img in os.listdir(obj_path)]

    if args.out:
        output_base = args.out
    else:
        output_base = os.path.dirname(__file__)

    generate_data(syn_img, images, args.num, args.start, output_base)

if __name__ == "__main__":
    main()
