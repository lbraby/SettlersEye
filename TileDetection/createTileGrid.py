#!/usr/bin/env python3

from ultralytics import YOLO
import argparse
import cv2
import math
import numpy as np

# train model: yolo task=detect mode=train epochs=20 data=C:\Users\lbrab\OneDrive\Documents\Programming\ComputerVisionFA23\SettlersEye\TileDetection\data\data.yaml model=yolov8m.pt imgsz=640

class Tile:
    # vertex1: top left vertex
    # vertex2: bottom right vertex
    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.boxWidth = vertex2[0] - vertex1[0]
        self.boxHeight = vertex2[1] - vertex1[1]
        self.center = ((vertex2[0]+vertex1[0])//2, (vertex2[1]+vertex1[1])//2)
        self.area = (vertex2[0]-vertex1[0])*(vertex2[1]-vertex1[1])
        self.neighbors = set()
        self.vertices = []

    def __str__(self):
        return f"Tile(vertex1={self.vertex1}, vertex2={self.vertex2}, center={self.center}, area={self.area})"

def build_parser():
    parser = argparse.ArgumentParser(description="Python wrapper to run YOLOv8 model on images")
    parser.add_argument("--image", help="Image to classify", required=True)
    parser.add_argument("--image_size", help="Specify image size", required=False, default=640)
    parser.add_argument("--confidence", help="Specify image size", required=False, default=.90)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    model = YOLO('yolov8m_tiledetection.pt')

    results = model.predict(args.image, imgsz=args.image_size, conf=args.confidence)[0]
    boxes = [(math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])) for box in results.boxes.xyxy.cpu().numpy()]
    if len(boxes) < 19:
        print("failed to detect full catan board\n")
        return
    
    image = cv2.imread(args.image)
    tiles = []
    for i in range(min(len(boxes), 19)): # detect at most 19 tiles
        # snippet = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
        tiles.append(Tile((boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3])))

        # image = cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (36,255,12), 1)
        # cv2.putText(image, f"{tiles[-1].area}", (boxes[i][0], boxes[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 3)

    tiles.sort(key = lambda x: (x.center[1], x.center[0]))
    meanWidth = np.mean([tile.boxWidth for tile in tiles])
    for i in range(len(tiles)):
        for j in range(i+1, len(tiles)):
            if math.dist(tiles[i].center, tiles[j].center) <= 1.35*meanWidth: # add neighbors to list
                tiles[i].neighbors.add(j)
                tiles[j].neighbors.add(i)
    
    for i in range(len(tiles)): # deduce vertices of hexagonal tiles
        neighbor = tiles[list(tiles[i].neighbors)[0]]

        # cos(30) = apothem/radius
        apothem = math.ceil(math.dist(tiles[i].center, neighbor.center))/2
        radius = apothem / 0.866025
        angleIncrement = 1.047197 # 2*PI/6
        angle = math.atan2(neighbor.center[1] - tiles[i].center[1], neighbor.center[0] - tiles[i].center[0])
        for _ in range(6):
            angle += angleIncrement
            tiles[i].vertices.append([tiles[i].center[0] + radius * math.sin(angle), tiles[i].center[1] + radius * math.cos(angle)])
        pts = np.array(tiles[i].vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (36,255,12), 2)

        # debug: draw bounding box and show number of neighbors
        # image = cv2.rectangle(image, (tiles[i].vertex1[0], tiles[i].vertex1[1]), (tiles[i].vertex2[0], tiles[i].vertex2[1]), (36,255,12), 2)
        cv2.putText(image, f"{len(tiles[i].neighbors)}", (tiles[i].center[0], tiles[i].center[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 3)

    cv2.imshow("detection", cv2.resize(image, None, fx=.25, fy=.25))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
