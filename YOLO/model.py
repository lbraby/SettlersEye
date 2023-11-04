#!/usr/bin/env python3

from ultralytics import YOLO
import argparse
from supervision import Detections, BoxAnnotator, ColorPalette
import cv2

def build_parser():
    parser = argparse.ArgumentParser(description="Python wrapper to run YOLOv8 model on images")
    parser.add_argument("--image", help="Image to classify", required=True)
    parser.add_argument("--model", help="Model used in YOLO detection", default="med_model.pt")
    parser.add_argument("--image_size", help="Specify image size", required=False, default=640)
    parser.add_argument("--confidence", help="Specify image size", required=False, default=.35)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    model = YOLO(args.model)
    box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=4, text_thickness=4, text_scale=2)
    results = model.predict(args.image, imgsz=args.image_size, conf=args.confidence)[0]
    detections = Detections(xyxy=results.boxes.xyxy.cpu().numpy(),
                            confidence=results.boxes.conf.cpu().numpy(),
                            class_id=results.boxes.cls.cpu().numpy().astype(int)
                            )

    frame = box_annotator.annotate(scene=results.orig_img, detections=detections)
    print(results)
    print(results.boxes)
    cv2.imshow("Annotated Board", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
