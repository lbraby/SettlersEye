#!/usr/bin/env python3

from ultralytics import YOLO
import argparse
import cv2
import math

# train model: yolo task=detect mode=train epochs=20 data=C:\Users\lbrab\OneDrive\Documents\Programming\ComputerVisionFA23\SettlersEye\TileDetection\data\data.yaml model=yolov8m.pt imgsz=640

def build_parser():
    parser = argparse.ArgumentParser(description="Python wrapper to run YOLOv8 model on images")
    parser.add_argument("--image", help="Image to classify", required=True)
    parser.add_argument("--image_size", help="Specify image size", required=False, default=640)
    parser.add_argument("--confidence", help="Specify image size", required=False, default=.90)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    model = YOLO('yolov8m_tiledetection.pt')

    results = model.predict(args.image, imgsz=args.image_size, conf=args.confidence)[0]
    boxes = [(math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])) for box in results.boxes.xyxy.cpu().numpy()]
    confidences = results.boxes.conf.cpu().numpy()
    
    image = cv2.imread(args.image)
    for i in range(min(len(boxes), 19)): # detect at most 19 tiles
        image = cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (36,255,12), 1)
        cv2.putText(image, "Tile", (boxes[i][0], boxes[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

    cv2.imshow("detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
