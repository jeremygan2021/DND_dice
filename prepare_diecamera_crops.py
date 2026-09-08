"""Validate dieCamera crops and create reproducible Ultralytics classify splits.

The source repository's holdout flag is authoritative.  Images are linked rather
than copied, so the immutable download remains the single source of truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2


VALID_TYPES = {"d4": range(1, 5), "d6": range(1, 7), "d8": range(1, 9),
               "d10": range(0, 10), "d12": range(1, 13), "d20": range(1, 21)}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace {destination}")
        return
    destination.symlink_to(os.path.relpath(source, destination.parent))


def prepare(source: Path, output: Path) -> dict:
    data_dir = source / "data"
    metadata = data_dir / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata}")

    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    names, hashes = set(), defaultdict(list)
    counts, source_counts = Counter(), Counter()
    failures = []

    for row in rows:
        name = row.get("file_name")
        die_type, value = row.get("type"), row.get("value")
        if not name or name in names:
            failures.append(f"duplicate/empty file_name: {name}")
            continue
        names.add(name)
        if die_type not in VALID_TYPES or value not in VALID_TYPES[die_type]:
            failures.append(f"invalid label: {name} {die_type}_{value}")
            continue
        image_path = data_dir / name
        if not image_path.is_file():
            failures.append(f"missing image: {name}")
            continue
        image = cv2.imread(str(image_path))
        if image is None or min(image.shape[:2]) < 8:
            failures.append(f"unreadable/small image: {name}")
            continue
        digest = file_digest(image_path)
        split = "val" if bool(row.get("holdout")) else "train"
        hashes[digest].append((split, name))
        label = f"{die_type}_{value}"
        counts[(split, label)] += 1
        source_counts[(split, str(row.get("source", "unknown")))] += 1
        link(image_path, output / "all60" / split / label / name)
        if die_type == "d20":
            link(image_path, output / "d20" / split / str(value) / name)

    leakage = [items for items in hashes.values()
               if {split for split, _ in items} == {"train", "val"}]
    if leakage:
        failures.extend(f"duplicate crosses splits: {items}" for items in leakage)
    if failures:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join(map(str, failures[:50])))

    report = {
        "source": str(source.resolve()),
        "rows": len(rows),
        "classes": sorted({label for _, label in counts}),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
        "source_counts": {f"{split}/{name}": count
                          for (split, name), count in sorted(source_counts.items())},
        "sha256_duplicates": sum(len(items) - 1 for items in hashes.values() if len(items) > 1),
        "cross_split_duplicates": len(leakage),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dataset/diecamera-crops"))
    parser.add_argument("--output", type=Path, default=Path("dataset/diecamera-classify"))
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
