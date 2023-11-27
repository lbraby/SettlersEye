#!/usr/bin/env python3

import cv2
import numpy as np
#from skimage import measure
import matplotlib.pyplot as plt
import sys
import argparse
import os
import json


def create_mask(filepath: str):
    # Read the image into grayscale
    print(filepath)
    sample = cv2.imread(filepath)

    points = read_segmentation(filepath)

    l, w = sample.shape[:2]
    mask = np.full((sample.shape[0], sample.shape[1]), 0, dtype=np.uint8)

    cv2.drawContours(mask, [points], 0, (255, 255, 255), -1)

    print(type(mask))
    print(type(sample))
    print(mask.shape)
    print(sample.shape)
    print(mask.size == sample.size)
    mask = cv2.bitwise_and(sample, sample, mask=mask)
    print(mask.size)
    #mask_im = cv2.resize(mask, (l, w))
    #mask_im = cv2.resize(mask, (int(l/4), int(w/4)))


    index = len(filepath) - 1
    while filepath[index] != ".":
        index -= 1

    root = filepath[:index]
    ext = filepath[index:]
    path = root + "_mask" + ext
    print(path)

    print(cv2.imwrite(path, mask))

    #cv2.imshow("mask", mask_im)


    #cv2.waitKey()

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



'''
    # Convert the original image to HSV
    # and take H channel for further calculations
    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    sample_h = sample_hsv[:, :, 0]

    # Uncomment this to show the H channel of the image
    #sample_small = cv2.resize(sample_h, (640, 480))
    #print('H channel of the image')
    #cv2.imshow(sample_small)


    # *** TASK *** It's a good place to apply morphological operations to the "binary_image"
    # definition of a kernel (a.k.a. structuring element):
    im_floodfill = binary_image.copy()
    h, w = binary_image.shape[:2]
    mask = np.zeros((h+2,w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    binary_image = binary_image | im_floodfill_inv

    kernel = np.ones((19, 19),np.uint8)
    binary_image = cv2.erode(src=binary_image, kernel=kernel, iterations = 1)

    # Uncomment this to show image after morphological transformation
    #sample_small = cv2.resize(sample_res, (640, 480))
    #print('Image after morphological transformation',sample_small)
    #cv2.imshow(sample_small)

    # Find connected pixels and compose them into objects
    labels = measure.label(binary_image)

    # Calculate features for each object; since we want to differentiate
    # between circular and square shapes, the major and minor axes may be less helpful
    # so think what other features could be useful; we will use also the centroid
    # to annotate the final result
    properties = measure.regionprops(labels)



    # *** TASK *** Calculate features for each object:
    # - some geometrical feature 1 (dimension 1)
    # - some intensity/color-based feature 2 (dimension 2)
    features = np.zeros((len(properties), 2))
    for i in range(0, len(properties)):
        features[i, 0] = properties[i].perimeter
        min_row, min_col, max_row, max_col = properties[i].bbox
        single_object_hue = sample_h[min_row:max_row,min_col:max_col]

        # calculating the average intensity of each image patch
        features[i, 1] = np.mean(single_object_hue)



    # *** TASK *** Show our objects in the feature space
    plt.plot(features[:, 0],features[:, 1], 'ro')
    plt.xlabel('Feature 1: perimeter')
    plt.ylabel('Feature 2: hue')
    plt.show()



    # *** TASK *** Choose the threshold
    thrF1 = 250
    thrF2 = 50



    # It's time to classify, count and display the objects
    squares = 0
    blue_circles = 0
    red_circles = 0

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))



    # *** TASK *** Code your classification here (using calculated features):
    for i in range(0, len(properties)):
        if (features[i, 0] > thrF1 and features[i, 1] <= thrF2):
        squares = squares + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.g', markersize=15)

        if (features[i, 0] <= thrF1 and features[i, 1] > thrF2):
        blue_circles = blue_circles + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.b', markersize=15)

        if (features[i, 0] <= thrF1 and features[i, 1] <= thrF2):
        red_circles = red_circles + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.r', markersize=15)
    plt.show()



    # That's all! Let's display the result:
    # print("I found %d squares, %d blue donuts, and %d red donuts." % (squares, blue_circles, red_circles))
'''

def build_parser():
    parser = argparse.ArgumentParser(description="Python utility to get catan piece object masks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filename", help="Image to create mask of")
    group.add_argument("--dir", metavar="DIRECTORY", help="directory to make mask of all images in", required=False)

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
