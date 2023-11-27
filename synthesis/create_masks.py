#!/usr/bin/env python3

import cv2
import numpy as np
# from skimage import measure
# import matplotlib.pyplot as plt
# import sys
import argparse
import os
import json


def create_mask(filepath: str):
    sample = cv2.imread(filepath)

    points = read_segmentation(filepath)

    l, w = sample.shape[:2]
    mask = np.full((sample.shape[0], sample.shape[1]), 0, dtype=np.uint8)

    cv2.drawContours(mask, [points], 0, (255, 255, 255), -1)

    mask = cv2.bitwise_and(sample, sample, mask=mask)

    index = len(filepath) - 1
    while filepath[index] != ".":
        index -= 1

    root = filepath[:index]
    ext = filepath[index:]
    path = root + "_mask" + ext

    print(f"Successfully wrote {path}: ", cv2.imwrite(path, mask))

    # Uncomment to see image
    '''
    mask_im = cv2.resize(mask, (l//4, w//4))

    cv2.imshow("mask", mask_im)

    cv2.waitKey()
    '''


def read_segmentation(filepath: str) -> np.array:
    index = len(filepath) - 1
    while filepath[index] != ".":
        index -= 1

    json_path = filepath[:index] + '.json'

    with open(json_path) as json_file:
        seg = json.load(json_file)

    points = seg.get("shapes", [{}])[0].get("points")

    points = np.array(points, dtype=np.int64)

    return points


def build_parser():
    parser = argparse.ArgumentParser(description="Python utility to get catan piece object masks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filename", '--file', '-f', help="Image to create mask of")
    group.add_argument("--dir", '--directory', '-d', metavar="DIRECTORY", help="directory to make mask of all images in", required=False)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.filename:
        path = os.path.abspath(args.filename)
        create_mask(path)
    else:
        root = os.path.abspath(args.dir)
        for filename in os.listdir(root):
            if filename.endswith(".json") or "mask" in filename:
                continue

            path = os.path.join(root, filename)

            if os.path.isdir(path):
                continue

            create_mask(path)

if __name__ == "__main__":
    main()
