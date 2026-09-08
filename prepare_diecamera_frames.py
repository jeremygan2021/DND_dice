"""Convert dieCamera frames into YOLO detect format with the seven dice classes.

The dieCamera ``frames`` repo carries one ``metadata.jsonl`` with a list of
fractional bounding boxes per die per frame.  This script turns each row into
one ``.jpg + .txt`` pair under ``<output>/images/<split>`` and
``<output>/labels/<split>`` so Ultralytics can train a 7-class detector.

Splits are made by *rig* (camera model) so a model never evaluates on a camera
it has trained on — that's the failure mode the source repo's README warns
about.  Frames whose ``camera`` field is unknown go into the train split because
their rig is not a domain boundary the model can be expected to generalise
across.

Output layout::

    dataset/diecamera-detect/
        images/{train,val}/<rig>/<epoch>/<file>.jpg   # symlinks
        labels/{train,val}/<rig>/<epoch>/<file>.txt
        data.yaml                                    # 7 classes, train/val paths
        report.json                                  # counts, rig split, class counts
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASS_ORDER = ("d4", "d6", "d8", "d10", "d12", "d20", "d100")
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_ORDER)}

# Rig → split.  Each rig is its own visual domain (lens, mount, lighting, tray).
# Keeping rigs out of the val set mirrors the "split by rig" rule from the
# dataset card.  ``unknown-rig`` rows have no documented camera so they go to
# train; their value is the depth, not the rig identity.
RIG_SPLIT = {
    "hue-hd-camera-0c45-6341": "val",
    "hd-usb-camera-05a3-9520": "train",
    "hd-usb-camera-05a3-9520-top-down": "train",
    "android-webcam-18d1-4eed": "train",
    "nintendo-switch-camera-057e-206d": "train",
    "nintendo-switch-camera-057e-206d-top-down": "train",
    "triveni-s-iphone-2-camera": "train",
    "triveni-s-iphone-2-camera-top-down": "train",
    "unknown-rig": "train",
}


def split_for(rig: str) -> str:
    return RIG_SPLIT.get(rig, "train")


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    os.symlink(os.path.relpath(source, destination.parent), destination)


def write_yolo_labels(label_path: Path, dice: list[dict]) -> int:
    lines = []
    for die in dice:
        die_type = die.get("type", "").lower()
        if die_type not in CLASS_INDEX:
            continue
        box = die.get("box") or {}
        x, y, w, h = box.get("x"), box.get("y"), box.get("w"), box.get("h")
        if None in (x, y, w, h):
            continue
        if w <= 0 or h <= 0:
            continue
        # YOLO expects normalised centre + size, clipped to [0, 1].
        cx = min(1.0, max(0.0, x + w / 2))
        cy = min(1.0, max(0.0, y + h / 2))
        nw = min(1.0, max(0.0, w))
        nh = min(1.0, max(0.0, h))
        lines.append(f"{CLASS_INDEX[die_type]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def prepare(source: Path, output: Path) -> dict:
    metadata = source / "data" / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata}")
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]

    rig_counts: Counter = Counter()
    class_counts: Counter = Counter()
    per_split_class: defaultdict[str, Counter] = defaultdict(Counter)
    per_split_frames: Counter = Counter()
    skipped = []

    if output.exists():
        shutil.rmtree(output)

    for row in rows:
        rel = row["file_name"]
        rig = rel.split("/", 1)[0]
        split = split_for(rig)
        image = source / "data" / rel
        if not image.is_file():
            skipped.append(f"missing image: {rel}")
            continue
        dice = row.get("dice") or []
        if not dice:
            skipped.append(f"empty dice list: {rel}")
            continue
        labels = [d for d in dice if (d.get("type", "").lower()) in CLASS_INDEX
                  and (d.get("box") or {}).get("w", 0) > 0]
        if not labels:
            skipped.append(f"no recognised types in: {rel}")
            continue
        link(image, output / "images" / split / rel)
        kept = write_yolo_labels(output / "labels" / split / rel, dice)
        rig_counts[rig] += 1
        per_split_frames[split] += 1
        for d in labels:
            class_counts[d["type"].lower()] += 1
            per_split_class[split][d["type"].lower()] += 1

    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for name, i in CLASS_INDEX.items()},
    }
    (output / "data.yaml").write_text(
        "# generated by prepare_diecamera_frames.py\n"
        + yaml_dump(data_yaml)
    )

    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "rows": len(rows),
        "frames_used": sum(rig_counts.values()),
        "frames_skipped": len(skipped),
        "rig_to_split": RIG_SPLIT,
        "frames_per_rig": dict(rig_counts),
        "frames_per_split": dict(per_split_frames),
        "boxes_per_class": dict(class_counts),
        "boxes_per_split_class": {split: dict(counter) for split, counter in per_split_class.items()},
        "skipped_sample": skipped[:20],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def yaml_dump(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v if not isinstance(v, str) else v!r}")
        elif isinstance(value, str):
            lines.append(f"{key}: {value!r}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dataset/diecamera-frames"))
    parser.add_argument("--output", type=Path, default=Path("dataset/diecamera-detect"))
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
