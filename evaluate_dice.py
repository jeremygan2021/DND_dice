"""Save reproducible local image results; outputs are observations, not ground truth."""
import argparse
import json
import time
from pathlib import Path
import cv2
from dice_engine import analyze_image
from dice_detector import DICE_TYPES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('images', nargs='+', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/validation'))
    p.add_argument('--dice-type', choices=('unknown',)+DICE_TYPES, default='unknown')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    report = []
    for i, path in enumerate(a.images, 1):
        image = cv2.imread(str(path))
        if image is None:
            p.error(f'Cannot read image: {path}')
        start = time.perf_counter()
        annotated, dice = analyze_image(image, dice_type=a.dice_type)
        entry = dict(image=str(path.resolve()), seconds=round(time.perf_counter()-start, 3),
                     detected=len(dice), recognized=sum(d['value'] is not None for d in dice), dice=dice)
        report.append(entry)
        cv2.imwrite(str(a.output/f'sample-{i}.jpg'), annotated)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    (a.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
