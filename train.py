# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train SARNet.

SARNet = Ultralytics YOLO26 with two structure-aware modules:

* ``LoGStem`` - Laplacian-of-Gaussian edge-enhanced stem (replaces the plain strided stem).
* ``SDS``     - semantic-guided dual-path structure downsampling (replaces the strided conv
                used for the P3/P4/P5 stage downsampling).

Example:
    python train.py --data /path/to/dataset/data.yaml --scale n --epochs 300 --batch 8
"""

import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_YAML = ROOT / "ultralytics/cfg/models/improve/my_module/yolo26-LoG-SDS.yaml"


def build_model(model_yaml: Path, scale: str, out_dir: Path) -> YOLO:
    """Build SARNet at the requested scale.

    Ultralytics infers the model scale from the yaml file name (``yolo26m.yaml`` -> ``m``).
    The SARNet yaml carries no scale suffix, so a scale-suffixed copy is written into the run
    directory (which also documents the exact architecture/scale used for the run) and loaded.

    Args:
        model_yaml (Path): SARNet model yaml.
        scale (str): One of n / s / m / l / x.
        out_dir (Path): Directory that receives the scale-suffixed yaml copy.

    Returns:
        (YOLO): Model ready for ``.train()``.
    """
    cfg = yaml.safe_load(Path(model_yaml).read_text(encoding="utf-8"))
    cfg["scale"] = scale
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"yolo26{scale}-LoG-SDS.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return YOLO(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SARNet (YOLO26 + LoGStem + SDS)")
    parser.add_argument("--model", type=str, default=str(MODEL_YAML), help="model yaml path")
    parser.add_argument("--scale", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="model scale")
    parser.add_argument("--data", type=str, required=True, help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="train/val image size")
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch", type=int, default=8, help="batch size (-1 for auto batch)")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--close-mosaic", type=int, default=0, help="disable mosaic for the last N epochs")
    parser.add_argument("--pretrained", type=str, default="", help="optional pretrained weights to initialize from")
    parser.add_argument("--project", type=str, default="runs/sarnet", help="save directory")
    parser.add_argument("--name", type=str, default="train", help="run name")
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_model(Path(args.model), args.scale, Path(args.project) / args.name)

    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        close_mosaic=args.close_mosaic,
        pretrained=bool(args.pretrained),
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
