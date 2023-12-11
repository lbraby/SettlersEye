#!/usr/bin/env python3

from ultralytics import YOLO
from ultralytics.engine import results
import argparse
from supervision import Detections, BoxAnnotator, ColorPalette
import cv2
from collections import defaultdict
from typing import Dict, Tuple, List

def build_parser():
    parser = argparse.ArgumentParser(description="Python wrapper to run YOLOv8 model on images")
    parser.add_argument("--image", help="Image to classify", required=True)
    parser.add_argument("--model", help="Model used in YOLO detection", default="synthetic_model.pt")
    parser.add_argument("--image_size", help="Specify image size", type=int, required=False, default=640)
    parser.add_argument("--confidence", help="Specify image size", type=float, required=False, default=.4)

    return parser

def check_piece_bounds(center: list, bboxes: list):
    for bbox in bboxes:
        if in_bounds(center, bbox):
            return False

    return True

def in_bounds(center: list, bbox: list):
    if center[0] >= bbox[0] and center[1] >= bbox[1] and center[0] >= bbox[2] and center[1] >= bbox[3]:
        return True

    return False

def tabulate_pieces(m_result: results):
    translation = m_result.names
    box_results = m_result.boxes
    bboxes = []

    scores: Dict[str, Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))

    for index, identification in enumerate(box_results.cls.numpy().astype(int)):
        color, piece = translation[identification].split("-")

        if in_bounds(box_results.xywh[index][:2], box_results.xyxy[index]):
            continue
        else:
            bboxes.append(box_results.xyxy[index])

        scores[color][piece].append(box_results.xywh[index][:2])

    return scores



def main():
    parser = build_parser()
    args = parser.parse_args()
    model = YOLO(args.model)
    box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=4, text_thickness=4, text_scale=2)
    results = model.predict(args.image, imgsz=args.image_size, conf=args.confidence)[0]
    detections = Detections(xyxy=results.boxes.xyxy.cpu().numpy(),
                            confidence=results.boxes.conf.cpu().numpy(),
                            class_id=results.boxes.cls.cpu().numpy().astype(int)
                            )

    frame = box_annotator.annotate(scene=results.orig_img, detections=detections)
    frame = cv2.resize(frame, (640, 640))
    scores = tabulate_pieces(results)
    print(scores)
    cv2.imshow("Annotated Board", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
