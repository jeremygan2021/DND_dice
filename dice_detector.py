"""Detector dispatcher:cv / ultralytics .pt / onnx 三选一。

- cv  :纯 OpenCV 轮廓检测 (无 dice_type)
- yolo :ultralytics + 自训 .pt (需 DICE_MODEL=...pt)
- onnx :G-G-Games/diecamera-models 三模型管线 + CUDA 加速 (需要 models/diecamera-models/*.onnx)

backend 由 DICE_DETECTOR=auto|cv|yolo|onnx 选定;运行时可通过 detector.set_mode() 切换。
"""
import os
from contextvars import ContextVar

request_detector_mode = ContextVar("request_detector_mode", default=None)
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
        env_mode = os.getenv('DICE_DETECTOR', 'auto').lower()
        if env_mode not in ('auto', 'cv', 'yolo', 'onnx'):
            raise ValueError('DICE_DETECTOR must be auto, cv, yolo or onnx')
        self._env_mode = env_mode
        self._mode_override = None  # 运行时 set_mode() 设进来
        self.model = None
        self.onnx_engine = None
        self.lock = threading.Lock()
        self.device = 'cpu'
        self._cv_warning = 'CV 无法可靠区分骰型；需要七类骰子训练权重'

    # ------------------------------------------------------------------
    def set_mode(self, mode):
        """运行时切换 backend (auto/cv/yolo/onnx)。'auto' 回到环境变量决定。"""
        if mode not in ('auto', 'cv', 'yolo', 'onnx'):
            raise ValueError('mode must be auto/cv/yolo/onnx')
        with self.lock:
            self._mode_override = None if mode == 'auto' else mode
            # 切换后下次 detect() 会按新 backend 走

    def _active_mode(self) -> str:
        m = request_detector_mode.get() or self._mode_override or self._env_mode
        if m == 'auto':
            onnx_dir = Path(__file__).parent / 'models' / 'diecamera-models'
            if (onnx_dir / 'dice-shape.onnx').exists():
                return 'onnx'
            if self.path.is_file():
                return 'yolo'
            return 'cv'
        return m

    @property
    def backend(self):
        return self._active_mode()

    def status(self):
        m = self.backend
        info = dict(backend=m,
                    yolo_weights_available=self.path.is_file(),
                    onnx_available=(Path(__file__).parent/'models'/'diecamera-models'/'dice-shape.onnx').exists(),
                    loaded=self.model is not None if m == 'yolo' else (self.onnx_engine is not None if m == 'onnx' else True),
                    device=self.device,
                    warning=self._cv_warning if m == 'cv' else ("ONNX 权重不含 D100；混合百分位骰需专门训练，或手动选 D100 使用 OCR" if m == "onnx" else None))
        return info

    def detect(self, image, fast=True):
        m = self._active_mode()
        if m == 'cv':
            from dice_engine import detect_dice_boxes
            return [dict(bbox=b, dice_type='unknown', confidence=0.0)
                    for b in detect_dice_boxes(image, fast=False)]
        if m == 'onnx':
            return self._detect_onnx(image)
        # yolo
        return self._detect_yolo(image)

    # ------------------------------------------------------------------
    def _detect_onnx(self, image):
        if self.onnx_engine is None:
            from dice_onnx_engine import engine as _eng
            self.onnx_engine = _eng
            self.device = _eng.device
        result = self.onnx_engine.detect_shapes(image)
        self.device = self.onnx_engine.device
        return result

    def _detect_yolo(self, image):
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

