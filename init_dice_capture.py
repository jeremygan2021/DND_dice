"""Create folders for collecting this project's own dice training images.

Supports the seven dice types the project trains:
    d4 d6 d8 d10 d12 d20 d100.

The crops split holds single-die images bucketed by ``<type>_<value>`` for the
value classifier. The frames split holds full dice-tray images plus matching
YOLO labels for the type detector. The incoming bucket is the staging area for
raw captures before they are cropped/annotated and moved into a split.
"""
from pathlib import Path


ROOT = Path("dataset/custom-dice")
LABELS = {
    "d4": [str(i) for i in range(1, 5)],
    "d6": [str(i) for i in range(1, 7)],
    "d8": [str(i) for i in range(1, 9)],
    "d10": [str(i) for i in range(0, 10)],
    "d12": [str(i) for i in range(1, 13)],
    "d20": [str(i) for i in range(1, 21)],
    "d100": [f"{i:02d}" for i in range(0, 100, 10)],
}
SPLITS = ("train", "val", "test")


def main() -> None:
    created = []
    for split in SPLITS:
        for die_type, values in LABELS.items():
            for value in values:
                bucket = ROOT / "crops" / split / f"{die_type}_{value}"
                bucket.mkdir(parents=True, exist_ok=True)
                created.append(bucket)
        for sub in ("images", "labels"):
            (ROOT / "frames" / split / sub).mkdir(parents=True, exist_ok=True)
            created.append(ROOT / "frames" / split / sub)
    (ROOT / "incoming" / "images").mkdir(parents=True, exist_ok=True)
    created.append(ROOT / "incoming" / "images")
    print(f"Created {len(created)} capture paths under {ROOT.resolve()}")


if __name__ == "__main__":
    main()
