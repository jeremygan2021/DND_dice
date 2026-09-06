"""Dice detection, serialized tracking and asynchronous conservative OCR.

Detection delegates to a locally trained YOLO model when available, otherwise
uses CV candidates. Values are invalidated on motion and read only after settling.
See README.md for D4 apex rules, percentile semantics and validation limitations.
"""
from __future__ import annotations

import logging
import re
from dice_detector import detector, suppress
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("yolo-web.dice")

# ----------------------------------------------------------------------
# 参数(可按部署机器微调)
# ----------------------------------------------------------------------
DETECT_MAX_SIDE = 720          # 检测用图最长边(越小越快,骰子太小时调大)
DETECT_MIN_SIDE_PX = 24        # 降采样图上骰子最小边长(相对 720 边)
FRAME_DIFF_GATE = 9.0          # 全图帧差门控:小于则复用上帧 bbox(0-255 均值)
ROI_DIFF_STILL = 12.0          # ROI 内容帧差阈值:小于视为"没在动"(0-255 均值)
STILL_MOVE_PX = 5.0            # bbox 中心位移(原图像素)低于此且 ROI 稳定 → 静止
MIN_STILL_FRAMES = 4           # 连续静止多少帧判 settled
TRACK_PURGE_S = 1.5            # 跟踪对象连续多少秒没被匹配就删除
TRACK_PURGE_MISS = 4           # 或连续丢失这么多帧(≈0.5s)直接删
OCR_COOLDOWN_S = 1.5           # 同 track 两次 OCR 最小间隔
OCR_MAX_FAILS = 5              # 连续失败这么多次后暂停,等骰子再动恢复
ROI_FP_CHANGE = 12.0           # 已识别 track 的 ROI 指纹差超过此 → 判定骰子被换/翻面
EMA_ALPHA = 0.55               # bbox EMA 平滑系数
ZERO_AS_TEN = True             # d10 "0" → 10
PERCENTILE_00_AS_100 = False   # percentile "00" → 100(默认 0)

# OCR 阈值与预算
OCR_HIGH_CONF = 0.60        # 命中即返回
OCR_ACCEPT_CONF = 0.30      # 预算用尽后的接受阈值
MAX_OCR_CANDS = 28          # 单颗骰子最多跑的候选张数(约 1s 预算)

# OCR 合法值域:常规数字面骰为 1..99(d4~d24、d100 的 10~90 都在内);
# d10 的 "0" → 10、percentile 的 "00" → 0/100 单独处理。
def _norm_text(t: str) -> str:
    """清洗 OCR 文本:只保留数字,纠正 O→0 / l、I→1 误读。"""
    t = t.strip().upper()
    t = t.replace("O", "0").replace("L", "1").replace("I", "1")
    return t if re.fullmatch(r"[0-9]{1,2}", t) else ""


def text_to_value(text: str, dice_type: str = "unknown") -> Tuple[Optional[int], str]:
    t = _norm_text(text)
    if not t:
        return None, ""
    v = int(t)
    if dice_type == "d100":
        return (v, t) if len(t) == 2 and v % 10 == 0 else (None, "")
    if dice_type == "d10" and t == "0":
        return 10, t
    if dice_type in ("d4", "d6", "d8", "d10", "d12", "d20"):
        return (v, t) if 1 <= v <= int(dice_type[1:]) and not t.startswith("0") else (None, "")
    if len(t) == 2 and t.startswith("0") and t != "00":
        return None, ""
    # Unknown geometry cannot establish whether a single zero means ten.
    if t == "00" or (1 <= v <= 20) or (v % 10 == 0 and 10 <= v <= 90):
        return v, t
    return None, ""


# ----------------------------------------------------------------------
# 阶段 A-1:快速检测(纯 CV,可在降采样图上跑)
# ----------------------------------------------------------------------
def detect_dice_boxes(img_bgr: np.ndarray, fast: bool = True) -> List[List[int]]:
    """检测多面骰 bbox [x1,y1,x2,y2](原图坐标)。

    两种模式都先降采样到 DETECT_MAX_SIDE 再检测(大图 4K 也能稳定检出),
    bbox 再映射回原图坐标。fast=False 加凸性/纹理校验,误检更少。
    """
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if fast:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        # 通道1:自适应阈值(局部明暗,抓骰子表面与底色的边缘)
        m1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 4)
        # 通道2:颜色距离局部均值(boxFilter 比 Gaussian 快)
        local_mean = cv2.boxFilter(small, ddepth=-1, ksize=(33, 33))
        color_diff = cv2.absdiff(small, local_mean)
        diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        _, m2 = cv2.threshold(diff_gray, 22, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(m1, m2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        min_area = (sw * sh) * 0.0025
        max_area = (sw * sh) * 0.28
        min_side = DETECT_MIN_SIDE_PX
        ratio_lo, ratio_hi = 0.45, 2.2
    else:
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        m1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 5)
        local_mean = cv2.boxFilter(small, ddepth=-1, ksize=(51, 51))
        color_diff = cv2.absdiff(small, local_mean)
        diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        _, m2 = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(blurred, 30, 90)
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        m3 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_big, iterations=2)
        combined = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        min_area = (sw * sh) * 0.003
        max_area = (sw * sh) * 0.22
        min_side = int(DETECT_MIN_SIDE_PX * 1.6)
        ratio_lo, ratio_hi = 0.5, 2.0

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Uniform dice tray: segment against robust border colour, independently of texture.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:,0], lab[:,-1]))
    background = np.median(border, axis=0)
    spread = np.median(np.linalg.norm(border-background, axis=1))
    if spread < 15:
        mask = (np.linalg.norm(lab-background, axis=2) > max(22, spread*3)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        extra, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) + list(extra)
    boxes: List[List[int]] = []
    inv = 1.0 / scale
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_side or bh < min_side:
            continue
        ratio = bw / max(bh, 1)
        if ratio < ratio_lo or ratio > ratio_hi:
            continue
        if True:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < 0.78:
                continue
        if area / (bw * bh) < 0.40 or x <= 1 or y <= 1 or x+bw >= sw-1 or y+bh >= sh-1:
            continue
        # 收缩一点,避免把骰子影子的外圈框进来
        shrink = 3 if fast else 6
        x1 = int((x + shrink) * inv)
        y1 = int((y + shrink) * inv)
        x2 = int((x + bw - shrink) * inv)
        y2 = int((y + bh - shrink) * inv)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return [d["bbox"] for d in suppress([dict(bbox=b, confidence=(b[2]-b[0])*(b[3]-b[1])) for b in boxes])]


def _frame_diff_ratio(frame_a_gray64: np.ndarray, frame_b_gray64: np.ndarray) -> float:
    return float(np.abs(frame_a_gray64.astype(np.int16)
                        - frame_b_gray64.astype(np.int16)).mean())


def _gray64(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA)


def _roi_fingerprint(img_bgr: np.ndarray, box) -> Optional[np.ndarray]:
    """ROI 内容指纹(24x24 灰度)。用于判断骰子是否被换面/翻动。"""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi, (24, 24), interpolation=cv2.INTER_AREA)


def _roi_diff(fp_a: Optional[np.ndarray], fp_b: Optional[np.ndarray]) -> float:
    if fp_a is None or fp_b is None or fp_a.shape != fp_b.shape:
        return 999.0
    return float(np.abs(fp_a.astype(np.int16) - fp_b.astype(np.int16)).mean())


# ----------------------------------------------------------------------
# 阶段 B:顶面梯形校正 + 二值化 + OCR
# ----------------------------------------------------------------------
def _upscale(base: np.ndarray, target: int = 340, min_s: float = 1.0,
             max_s: float = 3.0) -> np.ndarray:
    bh, bw = base.shape[:2]
    s = min(max_s, max(min_s, target / max(bh, bw)))
    if abs(s - 1.0) < 0.05:
        return base
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(base, (int(bw * s), int(bh * s)), interpolation=interp)


def _gray_and_enhanced(base: np.ndarray):
    """返回 (gray, enhanced, sat, rb, colorful?)"""
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    en = clahe.apply(gray)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    rb = cv2.absdiff(base[:, :, 2].astype(np.int16),
                     base[:, :, 0].astype(np.int16)).astype(np.uint8)
    colorful = bool(sat.mean() > 28.0 or rb.mean() > 24.0)
    return (gray, en, sat, rb, colorful)


def _ocr_candidates(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """Colour and enhanced images with 15-degree rotations for arbitrary dice orientation."""
    crop_up = _upscale(crop_bgr)
    h, w = crop_up.shape[:2]
    gray, enhanced, *_ = _gray_and_enhanced(crop_up)
    cands = [crop_up, enhanced]
    # Number glyphs are not restricted to right-angle rotations on polyhedral dice.
    for angle in range(15, 360, 15):
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        cands.append(cv2.warpAffine(crop_up, matrix, (w,h),
                                   borderMode=cv2.BORDER_REPLICATE))
    cands += [gray, cv2.bitwise_not(enhanced)]
    return cands


class FaceOCR:
    """单例:持有一个 RapidOCR 实例;所有调用串行(单 worker 线程池)。"""

    def __init__(self) -> None:
        self._reader = None
        self.providers = {}
        self._lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(16)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._api_executor: Optional[ThreadPoolExecutor] = None

    def _ensure(self):
        if self._reader is None:
            from rapidocr_onnxruntime import RapidOCR
            import inspect
            from pathlib import Path as P
            import yaml
            root = P(inspect.getfile(RapidOCR)).parent
            with open(root / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            from hardware import capabilities
            hw = capabilities()
            use_cuda = hw['cuda_available'] and 'CUDAExecutionProvider' in hw['onnx_available_providers']
            log.info("RapidOCR CUDA requested=%s", use_cuda)
            self._reader = RapidOCR(
                det_use_cuda=use_cuda,
                det_model_path=str(root / cfg["Det"]["model_path"]),
                cls_use_cuda=use_cuda,
                cls_model_path=str(root / cfg["Cls"]["model_path"]),
                rec_use_cuda=use_cuda,
                rec_model_path=str(root / cfg["Rec"]["model_path"]),
            )
            from ocr_compat import fix_legacy_scores
            self.legacy_score_fix = fix_legacy_scores(self._reader)
            for name in ('text_detector', 'text_cls', 'text_recognizer'):
                component = getattr(self._reader, name, None)
                session = getattr(getattr(component, 'infer', None), 'session', None)
                if session is None:
                    session = getattr(component, 'session', None)
                if not hasattr(session, 'get_providers'):
                    session = getattr(session, 'session', None)
                self.providers[name] = session.get_providers() if session is not None else []
        return self._reader

    def _submit(self, api, fn, *args, **kw):
        if not self._slots.acquire(blocking=False):
            return None
        try:
            with self._pool_lock:
                attr = '_api_executor' if api else '_executor'
                pool = getattr(self, attr)
                if pool is None:
                    pool = ThreadPoolExecutor(max_workers=2 if api else 1,
                                              thread_name_prefix='dice-api' if api else 'dice-ocr')
                    setattr(self, attr, pool)
                future = pool.submit(fn, *args, **kw)
            future.add_done_callback(lambda _: self._slots.release())
            return future
        except Exception:
            self._slots.release()
            raise

    def submit(self, fn, *args, **kw):
        return self._submit(False, fn, *args, **kw)

    def submit_api(self, fn, *args, **kw):
        return self._submit(True, fn, *args, **kw)

    def read_crop(self, crop_bgr: np.ndarray, dice_type="unknown") -> Tuple[Optional[int], float, str]:
        # Apex-numbered D4: require the repeated apex value on multiple visible faces.
        votes = {}
        with self._lock:
            reader = self._ensure()
            for cand in _ocr_candidates(crop_bgr)[:MAX_OCR_CANDS]:
                try:
                    res, _ = reader(cand)
                except Exception as e:
                    log.debug("OCR candidate error: %r", e)
                    continue
                valid = {}
                apex_boxes = []
                for item in res or []:
                    if len(item) < 3:
                        continue
                    v, norm = text_to_value(str(item[1]), dice_type)
                    confidence = float(item[2])
                    if v is not None and confidence >= .75:
                        points = np.asarray(item[0], dtype=float).reshape(-1, 2)
                        center = points.mean(axis=0)
                        ch, cw = cand.shape[:2]
                        if dice_type != "d4" and np.linalg.norm((center-[cw/2,ch/2])/[cw,ch]) > .24:
                            continue
                        if dice_type == "d4":
                            points = np.asarray(item[0], dtype=float).reshape(-1, 2)
                            center = points.mean(axis=0)
                            ch, cw = cand.shape[:2]
                            if np.linalg.norm((center - [cw/2, ch/2]) / [cw, ch]) > .35:
                                continue
                            if any(np.linalg.norm(center - old) < 8 for old in apex_boxes):
                                continue
                            apex_boxes.append(center)
                        valid[norm] = (v, confidence)
                # Several different visible numbers are ambiguous; never select the first side face.
                if len(valid) != 1 or (dice_type == "d4" and len(apex_boxes) < 2):
                    continue
                norm, (v, confidence) = next(iter(valid.items()))
                count, score, _ = votes.get(norm, (0, 0., v))
                votes[norm] = (count+1, score+confidence, v)
        ranked = sorted(votes.items(), key=lambda kv: kv[1][1], reverse=True)
        if ranked:
            norm, (count, score, v) = ranked[0]
            if count >= 2 and (len(ranked) == 1 or score >= ranked[1][1][1] * 1.5):
                return v, score/count, norm
        return None, 0.0, ""


# ----------------------------------------------------------------------
# 跟踪单元
# ----------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    box: List[int]                    # 平滑后 bbox [x1,y1,x2,y2]
    dice_type: str = "unknown"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    stable_frames: int = 0            # 连续静止帧数
    total_frames: int = 1
    center_px: Tuple[float, float] = (0.0, 0.0)
    # —— OCR 缓存 ——
    value: Optional[int] = None
    text: str = ""
    conf: float = 0.0
    source: str = "local"
    ocr_ts: float = 0.0
    ocr_pending: bool = False
    ocr_fails: int = 0                 # 连续失败次数(退避用)
    settled: bool = False
    # 用于"骰子被换面"检测的指纹(取稳定后一帧)
    ref_fp: Optional[np.ndarray] = None
    _prev_fp: Optional[np.ndarray] = None   # 上一帧 ROI 指纹(帧间运动判定)
    miss_frames: int = 0                    # 连续未被匹配的帧数
    gen: int = 0                            # 每次运动/换面 +1,作废在途 OCR 结果
    ocr_gave_up: bool = False               # 连续失败多次后暂停,等骰子再动才恢复

    @property
    def state(self) -> str:
        if self.settled:
            return "stable"
        if self.stable_frames > 0:
            return "settling"
        return "moving"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


def _center(box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _match_boxes(det_boxes, prev_centers, prev_boxes, prev_ids):
    """贪心匹配:新检测框 → 老 track_id。返回 {det_idx: track_id}。

    评分:IoU 为主;IoU 低于阈值但质心很近(快速移动帧之间 bbox 只部分重叠)
    时用距离给一个软分数,避免高速滚动时跟丢。
    """
    matched = {}
    used = set()          # prev 列表下标,防止一框吃多 track
    n_prev = len(prev_boxes)
    for di in range(len(det_boxes)):
        best_tid, best_score = None, 0.0
        cx, cy = _center(det_boxes[di])
        for ti in range(n_prev):
            if ti in used:
                continue
            pb = prev_boxes[ti]
            sc = _iou(det_boxes[di], pb)
            if sc >= 0.10:
                score = sc
            else:
                pcx, pcy = prev_centers[ti]
                diag = max(pb[2] - pb[0], pb[3] - pb[1], 1)
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                if dist < diag * 1.2 and sc > 0.02:
                    # 距离近的软匹配,分数上限低于 IoU 匹配,避免抢占
                    score = 0.08 * (1.0 - dist / (diag * 1.2))
                else:
                    continue
            if score > best_score:
                best_tid, best_score = prev_ids[ti], score
        if best_tid is not None:
            matched[di] = best_tid
            used.add(prev_ids.index(best_tid))
    return matched


# ----------------------------------------------------------------------
# 会话(DiceSession):服务端有状态,跨帧跟踪
# ----------------------------------------------------------------------
class DiceSession:
    def __init__(self, session_id: str, ocr: FaceOCR) -> None:
        self.sid = session_id
        self.ocr = ocr
        self.tracks: Dict[int, Track] = {}
        self._next_tid = 1
        self._prev_gray64: Optional[np.ndarray] = None
        self.last_access = time.time()
        self.prev_boxes: List[List[int]] = []
        self.frame_count = 0
        self.total_det_ms = 0.0
        self._write_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._last_detection = 0.0
        self._detections = []
        self._prev_full: Optional[Tuple[int, int]] = None
        self.expected_type = "unknown"
        self.api_mode: bool = False   # True: 稳定后的 OCR 走百炼 qwen-vl-ocr

    # ------------------------------------------------------------------
    def _update_tracks(self, boxes, img_bgr, ts):
        now = time.time()
        prev_tracks = list(self.tracks.values())
        prev_centers = [_center(t.box) for t in prev_tracks]
        prev_boxes = [t.box for t in prev_tracks]
        prev_ids = [t.track_id for t in prev_tracks]
        matched = _match_boxes(boxes, prev_centers, prev_boxes, prev_ids)

        used_tracks = set()
        new_boxes = []
        for di, box in enumerate(boxes):
            tid = matched.get(di)
            if tid is not None and tid in self.tracks:
                tr = self.tracks[tid]
                used_tracks.add(tid)
            else:
                tr = Track(self._next_tid, box)
                self._next_tid += 1
                self.tracks[tr.track_id] = tr
            detected_type = self._detections[di]["dice_type"] if di < len(self._detections) else "unknown"
            if self.expected_type != "unknown":
                detected_type = self.expected_type
            if detected_type != tr.dice_type:
                self._reset_value(tr)
                tr.dice_type = detected_type
            # EMA 平滑
            alpha = EMA_ALPHA if tr.total_frames > 1 else 1.0
            tr.box = [int(round(alpha * b + (1 - alpha) * ob))
                      for b, ob in zip(box, tr.box)]
            tr.last_seen = now
            tr.total_frames += 1
            tr.miss_frames = 0

            # —— 稳定性判定 ——
            cur_c = _center(box)
            move_px = ((cur_c[0] - tr.center_px[0]) ** 2 +
                       (cur_c[1] - tr.center_px[1]) ** 2) ** 0.5
            tr.center_px = cur_c
            fp = _roi_fingerprint(img_bgr, box)
            prev_fp = tr._prev_fp
            tr._prev_fp = fp

            # 换面检测:已识别的骰子,如果内容变了 → 缓存作废
            if (tr.value is not None and tr.ref_fp is not None
                    and _roi_diff(fp, tr.ref_fp) > ROI_FP_CHANGE):
                log.debug("track %d face changed -> reset value", tr.track_id)
                self._reset_value(tr)

            moving = move_px > STILL_MOVE_PX
            if fp is not None and prev_fp is not None and not moving:
                if _roi_diff(fp, prev_fp) > ROI_DIFF_STILL:
                    moving = True   # 位置没动但内容在变(原地转动/被拨动)

            if moving:
                self._reset_value(tr)
                tr.stable_frames = 0
                tr.settled = False
                tr.ocr_gave_up = False      # 骰子再动 → 恢复 OCR 重试资格
            else:
                tr.stable_frames += 1
                if tr.stable_frames >= MIN_STILL_FRAMES:
                    was_settled = tr.settled
                    tr.settled = True
                    if not was_settled:
                        log.debug("track %d settled (frames=%d)",
                                  tr.track_id, tr.stable_frames)
                    if tr.ref_fp is None:
                        tr.ref_fp = fp

            # —— 稳定且需要 OCR:后台触发 ——
            self._maybe_ocr(tr, img_bgr, ts)
            new_boxes.append(tr.box)

        # 处理没被匹配到的旧 track(检测丢失/刚离开画面):先衰减,超时才删
        for tr in self.tracks.values():
            if tr.last_seen == now:
                continue                       # 本帧被匹配过
            tr.miss_frames += 1
            self._reset_value(tr)
            if tr.miss_frames >= 2:
                tr.stable_frames = 0
                tr.settled = False
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if (now - tr.last_seen > TRACK_PURGE_S
                    or tr.miss_frames >= TRACK_PURGE_MISS):
                del self.tracks[tid]
        return new_boxes

    # ------------------------------------------------------------------
    def _reset_value(self, tr: Track):
        tr.value, tr.text, tr.conf = None, "", 0.0
        tr.settled = False
        tr.stable_frames = 0
        tr.ref_fp = None
        tr.ocr_fails = 0
        tr.ocr_gave_up = False
        tr.gen += 1   # 换面/值重置 → 作废在途 OCR

    # ------------------------------------------------------------------
    def _maybe_ocr(self, tr: Track, img_bgr: np.ndarray, ts: float):
        # 失败退避:连败 1/2/3+ 次 → 冷却 1.5/3/6 秒;连败 5 次放弃,
        # 直到骰子重新移动(_reset_value / moving)再试
        backoff = OCR_COOLDOWN_S * (2 ** min(tr.ocr_fails, 2))
        need = (tr.settled and tr.value is None and not tr.ocr_pending
                and not tr.ocr_gave_up and ts - tr.ocr_ts > backoff)
        if not need:
            return
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = tr.box
        pad = 10
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(w, x2 + pad), min(h, y2 + pad)
        crop = img_bgr[ya:yb, xa:xb]
        if crop.size == 0:
            return
        tr.ocr_pending = True
        tr.ocr_ts = ts
        crop = np.ascontiguousarray(crop).copy()
        tid = tr.track_id
        sid = self.sid
        use_api = self.api_mode
        gen_at_submit = tr.gen
        dice_type = tr.dice_type

        def _job():
            try:
                if use_api and dice_type != "d4":
                    v, conf, text = _ocr_per_die_api(crop)
                    v, text = text_to_value(text, dice_type)
                else:
                    v, conf, text = self.ocr.read_crop(crop, dice_type)
            except Exception:
                log.exception("OCR failed")
                v, conf, text = None, 0.0, ""
            with self._write_lock:
                tr = self.tracks.get(tid)
                if tr is None:
                    return
                tr.ocr_pending = False
                if v is None:
                    # 没读到 → 等冷却后重试;连败多轮则放弃直到骰子再动
                    if tr.settled and tr.gen == gen_at_submit:
                        tr.ocr_fails += 1
                        if tr.ocr_fails >= OCR_MAX_FAILS:
                            tr.ocr_gave_up = True
                            log.debug("sid=%s track %d OCR give up after %d fails",
                                      sid, tid, tr.ocr_fails)
                    return
                # OCR 期间骰子动过(gen 变)或已不处于稳定 → 丢弃过期结果
                if not tr.settled or tr.gen != gen_at_submit:
                    return
                tr.value, tr.conf, tr.text = v, conf, text
                tr.source = "api" if use_api and dice_type != "d4" else "local"
                tr.ocr_fails = 0
                log.debug("sid=%s track %d OCR(%s) ok: value=%s conf=%.2f",
                          sid, tid, tr.source, v, conf)

        future = self.ocr.submit_api(_job) if use_api else self.ocr.submit(_job)
        if future is None:
            tr.ocr_pending = False

    # ------------------------------------------------------------------
    def process_frame(self, img_bgr: np.ndarray, dice_type="unknown", use_api=False) -> Dict:
        with self._frame_lock, self._write_lock:
            if dice_type != self.expected_type or use_api != self.api_mode:
                for tr in self.tracks.values():
                    self._reset_value(tr)
            self.expected_type = dice_type
            self.api_mode = use_api
            return self._process_frame(img_bgr)

    def _process_frame(self, img_bgr: np.ndarray) -> Dict:
        """处理一帧。返回轻量响应 dict(bbox + 状态 + 缓存值)。"""
        ts = time.time()
        self.last_access = ts
        self.frame_count += 1
        t0 = time.perf_counter()

        gray64 = _gray64(img_bgr)
        diff = (_frame_diff_ratio(self._prev_gray64, gray64)
                if self._prev_gray64 is not None else 999.0)
        self._prev_gray64 = gray64

        # 门控:上一帧全部骰子已稳定 → 直接复用 bbox
        # (真正静止时才跳过检测;一旦有骰子在动,帧差或 settled 状态会让它失效。
        #  OCR 在途不影响:OCR 触发逻辑位于 _update_tracks,复用路径照常执行)
        prev_all_settled = bool(self.tracks) and all(
            t.settled for t in self.tracks.values())
        same_size = (self._prev_full is not None
                     and abs(self._prev_full[0] - img_bgr.shape[1]) < 2
                     and abs(self._prev_full[1] - img_bgr.shape[0]) < 2)
        reuse = (diff < FRAME_DIFF_GATE and prev_all_settled
                 and same_size and self.prev_boxes and ts-self._last_detection < .5)
        if reuse:
            boxes = [list(b) for b in self.prev_boxes]
            det_ms = 0.0
        else:
            t1 = time.perf_counter()
            self._detections = detector.detect(img_bgr, fast=True)
            boxes = [d["bbox"] for d in self._detections]
            self._last_detection = ts
            det_ms = (time.perf_counter() - t1) * 1000
        self._prev_full = (img_bgr.shape[1], img_bgr.shape[0])

        with self._write_lock:
            self.prev_boxes = self._update_tracks(boxes, img_bgr, ts)

        dice = []
        total = 0
        n_value = 0
        for tr in self.tracks.values():
            if tr.miss_frames or tr.total_frames < 3:
                continue
            d = {
                "track_id": tr.track_id,
                "dice_type": tr.dice_type,
                "bbox": [int(v) for v in tr.box],
                "value": tr.value,
                "text": tr.text,
                "conf": round(tr.conf, 3),
                "state": tr.state,
                "ocr_pending": tr.ocr_pending,
                "ocr_gave_up": tr.ocr_gave_up,
                "source": tr.source if tr.value is not None else None,
            }
            dice.append(d)
            if tr.settled and tr.value is not None:
                total += tr.value
                n_value += 1
        dice.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

        elapsed = (time.perf_counter() - t0) * 1000
        all_settled = (len(dice) > 0 and all(
            d["state"] == "stable" for d in dice))
        return {
            "num_dice": len(dice),
            "total_value": total,
            "n_recognized": n_value,
            "all_settled": all_settled,
            "det_ms": round(det_ms, 1),
            "elapsed_ms": round(elapsed, 1),
            "dice": dice,
            "frame": self.frame_count,
        }

    # ------------------------------------------------------------------
    def idle(self) -> bool:
        return time.time() - self.last_access > 30.0

    @property
    def needs_ocr(self) -> int:
        return sum(1 for t in self.tracks.values() if t.ocr_pending)


# ----------------------------------------------------------------------
# 会话注册表(单例)
# ----------------------------------------------------------------------
class DiceSessionRegistry:
    def __init__(self) -> None:
        self._sessions: Dict[str, DiceSession] = {}
        self._ocr = FaceOCR()
        self._lock = threading.Lock()

    def get(self, sid: str) -> DiceSession:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                s = DiceSession(sid, self._ocr)
                self._sessions[sid] = s
            return s

    def read_crop_sync(self, crop_bgr: np.ndarray, dice_type="unknown"):
        """同步 OCR(上传模式用,与摄像头后台 OCR 共用同一实例)。"""
        return self._ocr.read_crop(crop_bgr, dice_type)

    def cleanup(self):
        with self._lock:
            dead = [k for k, s in self._sessions.items() if s.idle()]
            for k in dead:
                del self._sessions[k]
            if dead:
                log.info("cleaned %d idle dice sessions", len(dead))


# 进程内单例
registry = DiceSessionRegistry()


# ----------------------------------------------------------------------
# 上传模式:整图一次性推理(同步,完整检测 + per-die OCR)
# ----------------------------------------------------------------------
def analyze_image(img_bgr: np.ndarray,
                  use_api: bool = False,
                  per_die_api: bool = False, dice_type: str = "unknown") -> Tuple[np.ndarray, List[Dict]]:
    """上传单张图完整识别。返回 (annotated_bgr, dice_info)。
    dice_info: [{bbox, value, text, conf, source}]
    """
    expected_type = dice_type
    detections = detector.detect(img_bgr, fast=False)
    detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

    dice_info: List[Dict] = []
    h, w = img_bgr.shape[:2]
    for detection in detections:
        box = detection["bbox"]
        dice_type = expected_type if expected_type != "unknown" else detection["dice_type"]
        x1, y1, x2, y2 = box
        pad = 10
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(w, x2 + pad), min(h, y2 + pad)
        crop = img_bgr[ya:yb, xa:xb]
        if use_api and per_die_api and dice_type != "d4":
            v, conf, text = _ocr_per_die_api(crop)
            v, text = text_to_value(text, dice_type)
            src = "api"
        else:
            v, conf, text = registry.read_crop_sync(crop, dice_type)
            src = "local"
        dice_info.append({"bbox": box, "value": v, "text": text,
                          "dice_type": dice_type, "detection_conf": detection["confidence"],
                          "conf": round(conf, 3), "source": src})

    annotated = _draw_boxes(img_bgr, dice_info)
    return annotated, dice_info


def _draw_boxes(img_bgr: np.ndarray, dice_info: List[Dict]) -> np.ndarray:
    out = img_bgr.copy()
    for d in dice_info:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 100), 2)
        label = str(d["value"]) if d["value"] is not None else "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 100), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    total = sum(d["value"] for d in dice_info if d["value"] is not None)
    header = f"Dice: {len(dice_info)}  Recognized sum: {total}"
    cv2.rectangle(out, (0, 0), (380, 36), (0, 0, 0), -1)
    cv2.putText(out, header, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _ocr_per_die_api(crop_bgr: np.ndarray):
    """单颗骰子裁图走百炼 qwen-vl-ocr(可选,最准最慢)。"""
    import subprocess, tempfile, os
    ch, cw = crop_bgr.shape[:2]
    scale = max(1.5, 240 / max(ch, cw))
    crop_big = cv2.resize(crop_bgr, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, crop_big)
        proc = subprocess.run(
            ["bl", "vision", "describe",
             "--image", tmp.name,
             "--prompt", "图中最显眼的数字(骰子顶面),只输出那个数字本身,不要任何解释。",
             "--model", "qwen-vl-ocr"],
            capture_output=True, text=True, timeout=30,
        )
        text = (proc.stdout or "").strip()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    v, norm = text_to_value(text)
    if v is not None:
        return v, 1.0, norm
    return None, 0.0, ""
