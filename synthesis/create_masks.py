#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import argparse
import os
import json
from typing import Optional


def create_mask(filepath: str, output_dir: Optional[str], crop_image: bool):
    sample = cv2.imread(filepath)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points = read_segmentation(filepath)

    l, w = sample.shape[:2]
    mask = np.full((sample.shape[0], sample.shape[1]), 0, dtype=np.uint8)

    cv2.drawContours(mask, [points], 0, (255, 255, 255), -1)
    mask_copy = mask.copy()

    mask = cv2.bitwise_and(sample, sample, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
    mask[:, :, 3] = mask_copy

    if crop_image:
        mins, maxes = find_bounds(points)
        mask = crop(mask, mins, maxes)

    index = len(filepath) - 1
    while index >= 0 and filepath[index] != ".":
        index -= 1

    if index <= 0:
        raise RuntimeError(f"Failed to find filetype of {filepath}, need this to find .json file")

    if not output_dir:
        root = filepath[:index]
        path = root + "_mask.png"
    else:
        root = os.path.basename(filepath)[:index-len(filepath)]
        path = os.path.join(output_dir, root + "_mask.png")

    print(f"Successfully wrote {path}: ", cv2.imwrite(path, mask))

    # Uncomment to see image

    '''
    mask_im = cv2.resize(mask, (l//4, w//4))

    cv2.imshow("mask", mask_im)

    cv2.waitKey()
    '''

def read_segmentation(filepath: str) -> np.array:
    index = len(filepath) - 1
    while index >= 0 and filepath[index] != ".":
        index -= 1

    if index <= 0:
        raise RuntimeError(f"Failed to find filetype of {filepath}, need this to find .json file")

    json_path = filepath[:index] + '.json'

    with open(json_path) as json_file:
        seg = json.load(json_file)

    points = seg.get("shapes", [{}])[0].get("points")

    points = np.array(points, dtype=np.int64)

    return points


def find_bounds(points: list) -> (tuple, tuple):
    x_min, y_min, x_max, y_max = (sys.maxsize, sys.maxsize, 0, 0)

    for point in points:
        if point[0] < x_min:
            x_min = point[0]
        if point[0] > x_max:
            x_max = point[0]
        if point[1] < y_min:
            y_min = point[1]
        if point[1] > y_max:
            y_max = point[1]

    return (x_min, y_min), (x_max, y_max)

def crop(image: np.array, mins: tuple, maxes: tuple) -> np.array:
    return image[mins[1]:maxes[1], mins[0]:maxes[0]]


def build_parser():
    parser = argparse.ArgumentParser(description="Python utility to get catan piece object masks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filename", '--file', '-f', help="Image to create mask of")
    group.add_argument("--dir", '--directory', '-d', metavar="DIRECTORY", help="directory to make mask of all images in", required=False)
    parser.add_argument("--out", '--output', '-o', metavar="OUTPUT DIR", help="directory to put output files", required=False)
    parser.add_argument("--crop", '-c', help="crop output files", required=False, action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.filename:
        path = os.path.abspath(args.filename)

        if not os.path.exists(path):
            print(f"{path} does not exist, skipping...")
            exit(1)

        try:
            create_mask(path, args.out, args.crop)
        except Exception as e:
            print(f"Failed to process {path}. Reason: {e}")
    else:
        root = os.path.abspath(args.dir)
        for filename in os.listdir(root):
            if filename.endswith(".json") or "mask" in filename:
                continue

            path = os.path.join(root, filename)

            if os.path.isdir(path):
                continue

            if not os.path.exists(path):
                print(f"{path} does not exist, skipping...")

            try:
                create_mask(path, args.out, args.crop)
            except Exception as e:
                print(f"Failed to process {path}. Reason: {e}")

if __name__ == "__main__":
    main()
