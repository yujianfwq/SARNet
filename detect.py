# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run inference with a trained SARNet checkpoint.

Examples:
    python detect.py --weights runs/sarnet/train/weights/best.pt --source ultralytics/assets/bus.jpg
    python detect.py --weights best.pt --source /path/to/images --save-txt
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="SARNet inference")
    parser.add_argument("--weights", type=str, required=True, help="trained checkpoint, e.g. best.pt")
    parser.add_argument("--source", type=str, required=True, help="image / directory / video / stream")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="cuda device i.e. 0 or cpu")
    parser.add_argument("--project", type=str, default="runs/sarnet", help="save directory")
    parser.add_argument("--name", type=str, default="predict", help="run name")
    parser.add_argument("--save-txt", action="store_true", help="save results as txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
