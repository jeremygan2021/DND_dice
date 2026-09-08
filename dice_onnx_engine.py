"""Local dieCamera ONNX inference with explicit model contracts.

The downloaded detector manifests declare YOLOX (objectness * class scores).
ConvNeXt uses ImageNet normalization and logits; Ultralytics whole-die
classification uses RGB 0..1 and already normalized probabilities.
The bundled checkpoints do not include percentile D100. Geometric model output
has no verified transform specification here and is deliberately not applied.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger("yolo-web.dice.onnx")

# ----------------------------------------------------------------------
# 常量（来自 G-G-Games/diecamera-models README）
# ----------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent / "models" / "diecamera-models"

# 合法骰子类型集合（按类型→最大面值映射）
TYPE_MAX = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d12": 12, "d20": 20}

# ImageNet 归一化（dice-value / dice-value-cls 都按 README 要这个）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SHAPE_CLASSES = ["d4", "d6", "d8", "d10", "d12", "d20"]
GLYPH_CLASSES = ["d10", "d12", "d20", "d4", "d8"]  # 注意顺序:d6 不在内
VALUE_CLASSES = [str(i) for i in range(21)]  # "0".."20"
CLS_CLASSES = (
    [f"d10_{i}" for i in range(10)]
    + [f"d12_{i}" for i in [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]]
    + [f"d20_{i}" for i in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9]]
    + [f"d4_{i}" for i in range(1, 5)]
    + [f"d6_{i}" for i in range(1, 7)]
    + [f"d8_{i}" for i in range(1, 9)]
)
# 简化：上面是 .classes.json 加载后的顺序(已经在文件里),我们直接读 json 不用硬编码

# 推理参数
SHAPE_CONF = float(os.getenv("DICE_ONNX_CONF", "0.35"))
GLYPH_CONF = float(os.getenv("DICE_ONNX_GLYPH_CONF", "0.30"))
VALUE_CONF = float(os.getenv("DICE_ONNX_VALUE_CONF", "0.65"))
NMS_IOU = 0.45
SHAPE_IMGSZ = 640


# ----------------------------------------------------------------------
# Letterbox:640×640,纵横比保持,114(灰)padding
# ----------------------------------------------------------------------
def letterbox(img: np.ndarray, new_size: int = SHAPE_IMGSZ, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_w, pad_h = new_size - nw, new_size - nh
    left, top = pad_w // 2, pad_h // 2
    if (h, w) != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if pad_w or pad_h:
        img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                                  cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top


def letterbox_to_chw(img_bgr: np.ndarray, new_size: int = SHAPE_IMGSZ) -> Tuple[np.ndarray, float, int, int]:
    """letterbox + BGR→RGB + 归一到 0..1 + NCHW。"""
    lb, r, l, t = letterbox(img_bgr, new_size)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None]  # 1×3×640×640
    return np.ascontiguousarray(chw), r, l, t


# ----------------------------------------------------------------------
# YOLOv8 输出解码:1×8400×(5+nc) → 每行 [x,y,w,h, obj_conf, cls_conf × nc]
# (这套模型看 README/快速测试,直接是裸输出,没 Sigmoid/Softmax)
# ----------------------------------------------------------------------
def _yolo_decode(out: np.ndarray, conf_thres: float, classes: List[str]):
    """out: (1, 8400, 5+nc) → 列表 [(x1,y1,x2,y2,score,cls_idx)] in 原图坐标, 含 letterbox 映射。

    YOLOv8 raw: 已经把 sigmoid 内嵌(opset >= 12 的 sigmoid 输出),
    所以 score = obj * cls_conf 直接用。
    """
    pred = np.asarray(out)[0]
    nc = len(classes)
    # Local checkpoints are YOLOX (5+nc). Also accept standard YOLOv8 (4+nc).
    if pred.shape[-1] not in (nc+4, nc+5) and pred.shape[0] in (nc+4, nc+5):
        pred = pred.T
    if pred.shape[-1] not in (nc+4, nc+5):
        raise ValueError(f"Unexpected detector output {out.shape} for {nc} classes")
    boxes = pred[:, :4]
    has_objectness = pred.shape[-1] == nc+5
    cls_conf = pred[:, 5:] if has_objectness else pred[:, 4:]
    cls_idx = cls_conf.argmax(axis=1)
    cls_score = cls_conf[np.arange(len(cls_idx)), cls_idx]
    score = cls_score * pred[:,4] if has_objectness else cls_score
    keep = score >= conf_thres
    boxes, score, cls_idx = boxes[keep], score[keep], cls_idx[keep]
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    # xywh → xyxy
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)
    return xyxy, score, cls_idx


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_thres: float):
    if len(xyxy) == 0:
        return []
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


# ----------------------------------------------------------------------
# 模型加载
# ----------------------------------------------------------------------
def _load_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model class manifest: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "classes" in data:
        return list(data["classes"])
    if not isinstance(data, list) or not data or not all(isinstance(c,str) for c in data):
        raise ValueError(f"Invalid model classes: {path}")
    return list(data)


class OnnxDiceEngine:
    """单例,持有 shape / glyph / value / cls 四个 InferenceSession。"""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self.sessions = {}
        self.shape_classes = _load_classes(model_dir / "dice-shape.classes.json") or SHAPE_CLASSES
        self.glyph_classes = _load_classes(model_dir / "dice-value-glyph.classes.json") or GLYPH_CLASSES
        self.value_classes = _load_classes(model_dir / "dice-value.classes.json") or VALUE_CLASSES
        self.cls_classes = _load_classes(model_dir / "dice-value-cls.classes.json") or CLS_CLASSES
        self.device = "cpu"
        self._loaded = False
        self._glyph_cache = {}
        self._glyph_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.model_dir / "dice-shape.onnx").exists()

    def _ensure(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import onnxruntime as ort
            available = ort.get_available_providers()
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(os.getenv("DICE_ONNX_THREADS", "2")))
            options.inter_op_num_threads = 1

            # 优先 CUDA; 但 CUDA 实际能不能跑由 dummy 推理决定(避免 torch.cuda
            # 检测不可靠导致静默退化)。先按注册顺序尝试,失败则降级。
            cuda_provider_options = {
                "device_id": 0,
                "gpu_mem_limit": int(os.getenv("DICE_ONNX_GPU_MEM_MB", "1024")) * 1024 * 1024,
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kNextPowerOfTwo",
            }

            chosen = None  # (providers, provider_options) 或 None
            if os.environ.get("DICE_ONNX_FORCE_CPU") == "1":
                chosen = (["CPUExecutionProvider"], None)
            elif "CUDAExecutionProvider" in available:
                chosen = (["CUDAExecutionProvider", "CPUExecutionProvider"],
                          [cuda_provider_options])
            else:
                chosen = (["CPUExecutionProvider"], None)

            for name in ("dice-shape", "dice-value-glyph", "dice-value", "dice-value-cls"):
                p = self.model_dir / f"{name}.onnx"
                if not p.exists():
                    log.warning("缺 %s,跳过", p)
                    continue
                providers, prov_opts = chosen
                # 实际尝试一次,确认 CUDA 真的能跑; 不行就降级
                sess = None
                if prov_opts is not None:
                    try:
                        sess = ort.InferenceSession(str(p),
                                                    providers=providers,
                                                    sess_options=options,
                                                    provider_options=prov_opts)
                        # 预热: 触发 cuBLAS 实际初始化
                        inp = sess.get_inputs()[0]
                        shape = [1 if isinstance(d, str) else d for d in inp.shape]
                        dummy = np.zeros(shape, dtype=np.float32)
                        sess.run(None, {inp.name: dummy})
                    except Exception as e:
                        log.warning("CUDA EP 实际跑 %s 失败: %r → 降级 CPU", name, e)
                        sess = None
                if sess is None:
                    sess = ort.InferenceSession(str(p),
                                                providers=["CPUExecutionProvider"],
                                                sess_options=options)
                self.sessions[name] = sess
                used = sess.get_providers()
                log.info("加载 ONNX: %s providers=%s", name, used)
            self.device = "cuda" if any("CUDAExecutionProvider" in s.get_providers() for s in self.sessions.values()) else "cpu"
            self._loaded = True

    # ------------------------------------------------------------------
    # 阶段 1:shape 检测
    # ------------------------------------------------------------------
    def detect_shapes(self, img_bgr: np.ndarray) -> List[dict]:
        """返回 [{bbox:[x1,y1,x2,y2], dice_type, confidence}] in 原图坐标。"""
        self._ensure()
        sess = self.sessions.get("dice-shape")
        if sess is None:
            return []
        chw, r, pad_l, pad_t = letterbox_to_chw(img_bgr, SHAPE_IMGSZ)
        out = sess.run(None, {sess.get_inputs()[0].name: chw})[0]
        xyxy, score, cls_idx = _yolo_decode(out, SHAPE_CONF, self.shape_classes)
        keep = _nms(xyxy, score, NMS_IOU)
        H, W = img_bgr.shape[:2]
        dets = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i]
            # 反 letterbox:从 640 框 → 原图
            x1 = (x1 - pad_l) / r
            y1 = (y1 - pad_t) / r
            x2 = (x2 - pad_l) / r
            y2 = (y2 - pad_t) / r
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "dice_type": str(self.shape_classes[int(cls_idx[i])]).lower(),
                "confidence": float(score[i]),
            })
        dets.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return dets

    # ------------------------------------------------------------------
    # 阶段 2:glyph 定位（在裁出的骰子框里跑；说明要求 "localises the numeral on the full frame"
    # 但为了与 shape bbox 配对,我们改为在 shape bbox + 0.1 margin 上跑,等价且更快）
    # ------------------------------------------------------------------
    def detect_glyphs(self, image):
        """Full-frame glyph inference, cached across dice from the same immutable frame."""
        self._ensure()
        key = (image.shape, hashlib.blake2b(np.ascontiguousarray(image), digest_size=16).digest())
        with self._glyph_lock:
            if key in self._glyph_cache:
                return self._glyph_cache[key]
            sess = self.sessions.get("dice-value-glyph")
            if sess is None:
                return []
            chw, r, pl, pt = letterbox_to_chw(image)
            out = sess.run(None, {sess.get_inputs()[0].name: chw})[0]
            boxes, scores, classes = _yolo_decode(out, GLYPH_CONF, self.glyph_classes)
            glyphs = []
            h,w = image.shape[:2]
            for i in _nms(boxes, scores, NMS_IOU):
                b = (boxes[i] - [pl,pt,pl,pt]) / r
                b = np.clip(b, [0,0,0,0], [w,h,w,h]).astype(int).tolist()
                if b[2] <= b[0] or b[3] <= b[1]:
                    continue
                glyphs.append(dict(bbox=b, dice_type=self.glyph_classes[int(classes[i])], confidence=float(scores[i])))
            if len(self._glyph_cache) >= 2:
                self._glyph_cache.pop(next(iter(self._glyph_cache)))
            self._glyph_cache[key] = glyphs
            return glyphs

    def detect_glyph(self, img_bgr, bbox, dice_type="unknown"):
        candidates = []
        x1,y1,x2,y2 = bbox
        for glyph in self.detect_glyphs(img_bgr):
            if dice_type != "unknown" and glyph['dice_type'] != dice_type:
                continue
            a,b,c,d = glyph['bbox']
            area = (c-a)*(d-b)
            overlap = max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))
            if area and overlap/area >= .85 and x1 <= (a+c)/2 <= x2 and y1 <= (b+d)/2 <= y2:
                candidates.append(glyph)
        candidates.sort(key=lambda g:g['confidence'], reverse=True)
        # Two similarly strong faces inside one die cannot establish its upward result.
        if not candidates or (len(candidates)>1 and candidates[1]['confidence'] > candidates[0]['confidence']*.8):
            return False, ()
        return True, tuple(candidates[0]['bbox'])

    # ------------------------------------------------------------------
    # 阶段 3a:value classifier:glyph 框 → 224×224 ImageNet → 21 logits
    # 门控到 dice_type 合法范围,softmax over in-range only
    # ------------------------------------------------------------------
    def read_value(self, img_bgr: np.ndarray, glyph_bbox, dice_type: str, glyph_already_localized: bool = True) -> Tuple[int, float, str]:
        """返回 (value, conf, text)。失败返回 (None, 0., "")。
        glyph_already_localized=True:glyph_bbox 是已定位的数字框,加 0.1 margin 后读
        glyph_already_localized=False:glyph_bbox 实际是骰子 bbox,直接读整骰 (whole-die path)
        """
        self._ensure()
        sess = self.sessions.get("dice-value")
        if sess is None or not glyph_bbox:
            return None, 0.0, ""
        h, w = img_bgr.shape[:2]
        gx1, gy1, gx2, gy2 = glyph_bbox
        gw, gh = gx2 - gx1, gy2 - gy1
        if glyph_already_localized:
            mx, my = int(gw * 0.1), int(gh * 0.1)
        else:
            mx, my = int(gw * 0.05), int(gh * 0.05)
        cxa = max(0, gx1 - mx)
        cya = max(0, gy1 - my)
        cxb = min(w, gx2 + mx)
        cyb = min(h, gy2 + my)
        crop = img_bgr[cya:cyb, cxa:cxb]
        if crop.size == 0:
            return None, 0.0, ""
        # 224×224 stretch-resized(README: "stretch-resized to 224×224, ImageNet-normalised")
        rs = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        logits = out[0]
        probs = _softmax(logits)
        bi = int(np.argmax(probs))
        text = self.value_classes[bi]
        v = _text_to_int(text, dice_type)
        # Preserve probability on the full class space. Renormalizing a wrong type
        # can otherwise turn an unsupported value into a confident legal number.
        margin = float(probs[bi] - np.partition(probs, -2)[-2])
        if v is None or probs[bi] < VALUE_CONF or margin < .15:
            return None, float(probs[bi]), text
        return v, float(probs[bi]), text

    # ------------------------------------------------------------------
    # 阶段 3b:cls classifier:dice crop 整图 → 60 类(type,face)
    # 主要用于 d6 (glyph 模型不含 d6)
    # ------------------------------------------------------------------
    def read_value_cls(self, img_bgr: np.ndarray, bbox, dice_type: str) -> Tuple[int, float, str]:
        """整骰子 crop → 60 路 (type_face)。返回 (value, conf, label)。
        若 dice_type 与预测类别的 type 不符,降级为 None。
        """
        self._ensure()
        sess = self.sessions.get("dice-value-cls")
        if sess is None:
            return None, 0.0, ""
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * 0.1), int(bh * 0.1)
        cxa = max(0, x1 - mx)
        cya = max(0, y1 - my)
        cxb = min(w, x2 + mx)
        cyb = min(h, y2 + my)
        crop = img_bgr[cya:cyb, cxa:cxb]
        if crop.size == 0:
            return None, 0.0, ""
        rs = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Ultralytics classification export expects RGB 0..1, unlike ConvNeXt.
        norm = rgb
        x = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        probs = _as_probabilities(out[0])
        bi = int(np.argmax(probs))
        label = self.cls_classes[bi]
        conf = float(probs[bi])
        # label 形如 "d10_0"
        if "_" not in label:
            return None, conf, label
        t, face_s = label.split("_", 1)
        try:
            v = int(face_s)
        except ValueError:
            return None, conf, label
        # 约束到 dice_type(若指定)
        if dice_type not in ("unknown", t):
            return None, conf, label
        margin = float(probs[bi] - np.partition(probs, -2)[-2])
        v = _text_to_int(face_s, t)
        if conf < VALUE_CONF or margin < .15:
            return None, conf, label
        return v, conf, face_s


    def read_die(self, image, bbox, dice_type):
        result = dict(value=None, conf=0., text="", source="onnx", reason="low_confidence", glyph_bbox=None)
        if dice_type == "d100":
            result['reason'] = 'unsupported_type'
            return result
        found, glyph = self.detect_glyph(image, bbox, dice_type)
        gv,gc,gt = self.read_value(image, glyph, dice_type) if found else (None,0.,"")
        cv,cc,ct = self.read_value_cls(image, bbox, dice_type)
        result['glyph_bbox'] = list(glyph) if found else None
        if gv is not None and cv is not None and gv != cv:
            result['reason'] = 'model_disagreement'
            return result
        if gv is not None or cv is not None:
            v,c,t = (gv,gc,gt) if gv is not None else (cv,cc,ct)
            result.update(value=v, conf=c, text=t, reason=None)
            return result
        # glyph + cls 都失败:再让 value 模型直接在整骰 crop 上跑(whole-die path),
        # README 描述的 "Two-pass" 管线在 glyph 不存在时同样可用,且 d6 通常赢.
        vv,vc,vt = self.read_value(image, bbox, dice_type, glyph_already_localized=False)
        if vv is not None:
            result.update(value=vv, conf=vc, text=vt, reason=None)
        elif not found:
            result['reason'] = 'glyph_not_found'
        return result


def _as_probabilities(output):
    output = np.asarray(output, dtype=np.float32)
    if np.all(np.isfinite(output)) and output.min() >= 0 and output.max() <= 1 and np.isclose(output.sum(),1,atol=1e-4):
        return output
    return _softmax(output)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _text_to_int(text: str, dice_type: str):
    """与 dice_engine.text_to_value 同语义,但只返回值。"""
    import re
    t = text.strip()
    if not re.fullmatch(r"\d{1,2}", t):
        return None
    v = int(t)
    if dice_type == "d10" and t == "0":
        return 10
    if dice_type in TYPE_MAX:
        mx = TYPE_MAX[dice_type]
        if 1 <= v <= mx and not t.startswith("0"):
            return v
        return None
    return None


# 进程内单例
engine = OnnxDiceEngine()
