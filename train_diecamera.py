"""Train D20-only or all-60 face classifiers, then optionally export ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("d20", "all60"), default="d20")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/diecamera-classify"))
    parser.add_argument("--model", default="yolo11s-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    data = (args.data_root / args.task).resolve()
    if not (data / "train").is_dir() or not (data / "val").is_dir():
        parser.error(f"Prepared dataset not found: {data}; run prepare_diecamera_crops.py")

    from ultralytics import YOLO
    model = YOLO(args.model)
    result = model.train(
        data=str(data), epochs=args.epochs, imgsz=224, batch=args.batch,
        device=args.device, workers=args.workers, patience=20,
        project="runs/diecamera", name=f"{args.task}-cls",
        degrees=180.0, fliplr=0.0, flipud=0.0,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    YOLO(str(best)).val(data=str(data), imgsz=224, device=args.device)
    if args.export:
        YOLO(str(best)).export(format="onnx", imgsz=224, simplify=True)


if __name__ == "__main__":
    main()
