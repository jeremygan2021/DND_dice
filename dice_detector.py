"""Optional trained seven-class YOLO detector. Never downloads generic weights."""
import os
import threading
from pathlib import Path

DICE_TYPES = ('d4', 'd6', 'd8', 'd10', 'd12', 'd20', 'd100')


def suppress(detections, threshold=0.45, limit=30):
    kept = []
    for item in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        a = item['bbox']
        aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        if not aa:
            continue
        duplicate = False
        for other in kept:
            b = other['bbox']
            bb = (b[2]-b[0]) * (b[3]-b[1])
            inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
            if inter / (aa+bb-inter) > threshold or inter/min(aa,bb) > 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
        if len(kept) >= limit:
            break
    return kept


class DiceDetector:
    def __init__(self):
        self.path = Path(os.getenv('DICE_MODEL', str(Path(__file__).parent/'models/dice.pt')))
        self.mode = os.getenv('DICE_DETECTOR', 'auto').lower()
        if self.mode not in ('auto', 'cv', 'yolo'):
            raise ValueError('DICE_DETECTOR must be auto, cv or yolo')
        self.model = None
        self.lock = threading.Lock()
        self.device = 'cpu'

    @property
    def backend(self):
        return 'yolo' if self.mode == 'yolo' or (self.mode == 'auto' and self.path.is_file()) else 'cv'

    def status(self):
        return dict(backend=self.backend, weights_available=self.path.is_file(),
                    loaded=self.model is not None, device=self.device,
                    warning='CV 无法可靠区分骰型；需要七类骰子训练权重' if self.backend == 'cv' else None)

    def detect(self, image, fast=True):
        if self.backend == 'cv':
            from dice_engine import detect_dice_boxes
            return [dict(bbox=b, dice_type='unknown', confidence=0.0)
                    for b in detect_dice_boxes(image, fast)]
        with self.lock:
            if self.model is None:
                if not self.path.is_file():
                    raise ValueError('缺少骰子专用权重: ' + str(self.path))
                import torch
                from ultralytics import YOLO
                self.device = os.getenv('DICE_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
                model = YOLO(str(self.path))
                if set(str(n).lower() for n in model.names.values()) != set(DICE_TYPES):
                    raise ValueError('骰子模型必须包含 d4,d6,d8,d10,d12,d20,d100 七类')
                self.model = model
            result = self.model.predict(image, conf=float(os.getenv('DICE_CONF', '.55')),
                iou=.45, agnostic_nms=True, max_det=30, imgsz=640,
                device=self.device, half=self.device.startswith('cuda'), verbose=False)[0]
            detections = []
            h,w = image.shape[:2]
            for b, c, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                x1,y1,x2,y2 = map(int,b)
                detections.append(dict(bbox=[max(0,x1),max(0,y1),min(w,x2),min(h,y2)],
                    dice_type=str(self.model.names[int(c)]).lower(), confidence=float(score)))
            return suppress(detections)


detector = DiceDetector()
