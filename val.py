# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate a trained SARNet checkpoint.

Example:
    python val.py --weights runs/sarnet/train/weights/best.pt --data /path/to/dataset/data.yaml
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SARNet")
    parser.add_argument("--weights", type=str, required=True, help="trained checkpoint, e.g. best.pt")
    parser.add_argument("--data", type=str, required=True, help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="val image size")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="cuda device i.e. 0 or cpu")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="dataset split to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )


if __name__ == "__main__":
    main()
