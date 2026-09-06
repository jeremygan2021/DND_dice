"""Train/validate locally supplied, annotated dice images (no implicit weight download)."""
import argparse
from pathlib import Path
from dice_detector import DICE_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path, help='Local .pt pretrained weights or .yaml architecture')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    import yaml
    from ultralytics import YOLO
    if not args.model.is_file() or not args.data.is_file():
        parser.error('data and model must be existing local files')
    data = yaml.safe_load(args.data.read_text())
    names = data.get('names', {})
    names = list(names.values()) if isinstance(names, dict) else names
    if set(names) != set(DICE_TYPES):
        parser.error('dataset must contain exactly the seven dice classes')
    model = YOLO(str(args.model))
    if args.validate:
        model.val(data=str(args.data.resolve()), imgsz=640, device=args.device, plots=True)
    else:
        model.train(data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch,
                    imgsz=640, device=args.device, workers=2, amp=args.device != 'cpu',
                    project='runs/dice', name='train', degrees=180, fliplr=0, flipud=0)


if __name__ == '__main__':
    main()
