# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export a trained SARNet checkpoint (ONNX / TensorRT / OpenVINO / ...).

Examples:
    python export.py --weights runs/sarnet/train/weights/best.pt --format onnx
    python export.py --weights best.pt --format engine --half
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="SARNet export")
    parser.add_argument("--weights", type=str, required=True, help="trained checkpoint, e.g. best.pt")
    parser.add_argument("--format", type=str, default="onnx", help="export format, e.g. onnx / engine / openvino")
    parser.add_argument("--imgsz", type=int, default=640, help="export image size")
    parser.add_argument("--batch", type=int, default=1, help="export batch size")
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--simplify", action="store_true", help="simplify the exported graph")
    parser.add_argument("--device", type=str, default="0", help="cuda device i.e. 0 or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.half,
        simplify=args.simplify,
        device=args.device,
    )


if __name__ == "__main__":
    main()
