"""
模型加载与推理。
- 模型权重从魔搭 ModelScope 拉到本地，不依赖 HuggingFace。
- doclayout 模型 ID：opendatalab/DocLayout-YOLO-DocStructBench（魔搭镜像）
- dice 模型：先用通用 YOLOv8n 作为占位（后续可用 Roboflow dice 数据集微调）
"""
from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

log = logging.getLogger("yolo-web.loader")

# 模型魔搭 ID（用户指定：不要从 HF 拉，用魔搭 URL）
MODELSPEC_DOCLAYOUT = "opendatalab/DocLayout-YOLO-DocStructBench"
# 通用 YOLOv8n 作为骰子检测占位（Ultralytics 官方也托管在 GitHub release，
# 不算 HF；后续用 Roboflow dice 数据集微调的权重也通过魔搭托管）
DICE_PLACEHOLDER_ID = "yolov8n"  # Ultralytics 自带的 nano 检测模型


class ModelLoader:
    """单例懒加载器：首次用到某个模型才下载/初始化。"""

    # DocLayout-YOLO 类别（10 类，来自 DocStructBench）
    DOCLAYOUT_CLASSES = [
        "title",           # 标题
        "plain_text",      # 正文段落
        "figure",          # 图
        "figure_caption",  # 图注
        "table",           # 表格
        "table_caption",   # 表注
        "header",          # 页眉
        "footer",          # 页脚
        "reference",       # 参考文献
        "equation",        # 公式
    ]

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._detect_device()
        self._doclayout = None
        self._dice = None
        self._local_ocr = None
        log.info(f"ModelLoader 初始化：device={self.device}, dir={self.model_dir}")

    # ------------------------------------------------------------------
    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "doclayout": self._doclayout is not None,
            "dice": self._dice is not None,
        }

    # ==================================================================
    # 文档版面 YOLO
    # ==================================================================
    def _ensure_doclayout(self) -> Path:
        """从魔搭拉 doclayout 权重到本地。"""
        target = self.model_dir / "doclayout_yolo.pt"
        if target.exists() and target.stat().st_size > 1_000_000:
            log.info(f"doclayout 模型已存在：{target}")
            return target

        log.info(f"从魔搭 ModelScope 下载 DocLayout-YOLO 权重 ...")
        # 强制从魔搭下载
        from modelscope import snapshot_download
        cache_dir = snapshot_download(
            MODELSPEC_DOCLAYOUT,
            cache_dir=str(self.model_dir.parent / ".ms_cache"),
        )
        # 找 .pt 文件
        pt_files = list(Path(cache_dir).rglob("*.pt"))
        if not pt_files:
            raise RuntimeError(f"魔搭包里没找到 .pt 权重：{cache_dir}")
        src = pt_files[0]
        shutil.copy2(src, target)
        log.info(f"doclayout 权重已落盘：{target} ({target.stat().st_size//1024//1024} MB)")
        return target

    def _load_doclayout(self):
        from ultralytics import YOLO
        weight = self._ensure_doclayout()
        log.info(f"加载 YOLO 模型：{weight}")
        self._doclayout = YOLO(str(weight))
        return self._doclayout

    def infer_doclayout(self, img_bgr: np.ndarray, conf: float = 0.25):
        """文档版面拉框。返回 (annotated_bgr, blocks)。"""
        if self._doclayout is None:
            self._load_doclayout()
        results = self._doclayout.predict(
            source=img_bgr, conf=conf, device=self.device, verbose=False
        )
        r = results[0]
        annotated = r.plot()  # BGR

        blocks: List[Dict[str, Any]] = []
        if r.boxes is not None:
            for box, cls_idx, score in zip(
                r.boxes.xyxy.cpu().numpy().astype(int),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box.tolist()
                cls_name = self.DOCLAYOUT_CLASSES[cls_idx] if cls_idx < len(self.DOCLAYOUT_CLASSES) else f"cls_{cls_idx}"
                blocks.append({
                    "class": cls_name,
                    "conf": round(float(score), 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
        # 按 y 坐标排序（视觉自上而下）
        blocks.sort(key=lambda b: b["bbox"][1])
        return annotated, blocks

    # ==================================================================
    # 骰子检测 + 阿里云百炼 OCR
    # ==================================================================
    def infer_dice(self, img_bgr: np.ndarray, conf: float = 0.3, use_api: bool = True, per_die_api: bool = True, fast: bool = False):
        """跑团多面骰检测（D4~D24）+ OCR 读顶面数字。
        fast=True：极速 CV（约 10-30ms/帧，摄像头模式）
        fast=False：完整多模态 CV + 校验（约 50-100ms/帧，上传模式）
        """
        annotated, bboxes = self._detect_dice_polyhedral(img_bgr, fast=fast)
        bboxes.sort(key=lambda b: (b[1], b[0]))
        if use_api:
            if per_die_api and bboxes:
                values = self._ocr_per_die_via_bailian(img_bgr, bboxes)
            else:
                values = self._ocr_dice_values_via_bailian(img_bgr, expected=len(bboxes))
        else:
            values = self._ocr_per_die_local(img_bgr, bboxes) if bboxes else []

        # 按 bbox 顺序构造 dice_info；OCR 失败（None）跳过，避免错位
        dice_info: List[Dict[str, Any]] = []
        for i, bbox in enumerate(bboxes):
            v = values[i] if i < len(values) else None
            if v is None:
                continue
            dice_info.append({
                "bbox": [int(x) for x in bbox],
                "value": int(v),
            })

        # 在 annotated 图上画标注
        for d in dice_info:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)
            label = f"{d['value']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 100), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 加总标注
        total = sum(d["value"] for d in dice_info)
        header = f"骰子数: {len(dice_info)}   总点数: {total}"
        cv2.rectangle(annotated, (0, 0), (380, 36), (0, 0, 0), -1)
        cv2.putText(annotated, header, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated, dice_info

    # ------------------------------------------------------------------
    def _get_local_ocr(self):
        """懒加载 RapidOCR（ONNX，本地 CPU/GPU，~10MB 模型首次会下）。"""
        if self._local_ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            import inspect
            # 读默认 config → 拿 model_path 注入（UpdateParameters 要求 model_path 必填）
            from pathlib import Path as P
            root = P(inspect.getfile(RapidOCR)).parent
            import yaml
            with open(root / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            use_cuda = False
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers() and self.device.startswith("cuda"):
                    use_cuda = True
            except ImportError:
                pass
            log.info(f"RapidOCR CUDA={'ON' if use_cuda else 'OFF'}")
            self._local_ocr = RapidOCR(
                det_use_cuda=use_cuda,
                det_model_path=str(root / cfg["Det"]["model_path"]),
                cls_use_cuda=use_cuda,
                cls_model_path=str(root / cfg["Cls"]["model_path"]),
                rec_use_cuda=use_cuda,
                rec_model_path=str(root / cfg["Rec"]["model_path"]),
            )
        return self._local_ocr

    # ------------------------------------------------------------------
    def _ocr_dice_values_local(self, img_bgr: np.ndarray, expected: int) -> List[int]:
        """RapidOCR 读全图所有骰子数字（1~24），按从左到右排序返回。
        摄像头模式用：单帧 100~300ms，无网络依赖。
        多极性 + CLAHE + 颜色通道，覆盖白字/黑字/金字/红字。
        """
        reader = self._get_local_ocr()
        h, w = img_bgr.shape[:2]
        if w > 1280:
            scale = 1280 / w
            img_small = cv2.resize(img_bgr, (1280, int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            img_small = img_bgr

        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]                       # 饱和度：金字/红字敏感
        rb_diff = cv2.absdiff(img_small[:, :, 2], img_small[:, :, 0])  # R-B：金色强信号

        candidates = [
            gray, cv2.bitwise_not(gray),
            enhanced, cv2.bitwise_not(enhanced),
            sat, cv2.bitwise_not(sat),
            rb_diff, cv2.bitwise_not(rb_diff),
        ]

        best: List[int] = []
        for img in candidates:
            try:
                result, _elapsed = reader(img)
            except Exception:
                continue
            if not result:
                continue
            items = []
            for item in result:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                box, text, conf = item[0], item[1], item[2]
                text = str(text).strip()
                if not text.isdigit():
                    continue
                v = int(text)
                if not (1 <= v <= 99):
                    continue
                try:
                    xs = [float(p[0]) for p in box]
                    cx = sum(xs) / len(xs)
                except Exception:
                    cx = 0
                items.append((cx, v, float(conf)))
            if not items:
                continue
            items.sort(key=lambda x: x[0])
            values = [v for _, v, _ in items]
            if expected:
                values = values[:expected]
            if len(values) > len(best):
                best = values
        return best

    # ------------------------------------------------------------------
    def _ocr_dice_values_via_bailian(self, img_bgr: np.ndarray, expected: int) -> List[int]:
        """调一次百炼 qwen-vl-ocr 读整张图所有骰子顶面数字，按从左到右/从上到下顺序返回。
        返回值长度 = min(检测到的 bbox 数, API 返回数字数)。
        """
        import subprocess, tempfile, re

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            cv2.imwrite(tmp.name, img_bgr)
            prompt = (
                "图中所有骰子顶面朝上的数字，请按从左到右、从上到下的顺序，"
                "只输出数字本身，多个数字用英文逗号分隔，例如：4,12,20。"
                "如果没有识别到任何骰子顶面数字，输出：无。"
            )
            proc = subprocess.run(
                ["bl", "vision", "describe",
                 "--image", tmp.name,
                 "--prompt", prompt,
                 "--model", "qwen-vl-ocr"],
                capture_output=True, text=True, timeout=60,
            )
            text = (proc.stdout or "").strip()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        if not text or text.startswith("无"):
            return []
        nums: List[int] = []
        for chunk in re.split(r"[,，\s]+", text):
            chunk = chunk.strip().rstrip(".。")
            if chunk.isdigit():
                v = int(chunk)
                if 1 <= v <= 99:
                    nums.append(v)
        if expected and len(nums) > expected:
            nums = nums[:expected]
        return nums

    # ------------------------------------------------------------------
    def _ocr_per_die_local(self, img_bgr: np.ndarray, bboxes: List[List[int]]) -> List[Optional[int]]:
        """本地 per-die OCR：裁剪 + 放大 + 多通道（优先 gray，命中即返回）。"""
        reader = self._get_local_ocr()
        H, W = img_bgr.shape[:2]
        results: List[Optional[int]] = []
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            pad = 14
            xa = max(0, x1 - pad); ya = max(0, y1 - pad)
            xb = min(W, x2 + pad); yb = min(H, y2 + pad)
            crop = img_bgr[ya:yb, xa:xb]
            ch, cw = crop.shape[:2]
            scale = max(2.5, 400.0 / max(ch, cw))
            crop_big = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop_big, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)
            hsv = cv2.cvtColor(crop_big, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            R, G, B = crop_big[:, :, 2], crop_big[:, :, 1], crop_big[:, :, 0]
            rb_diff = cv2.absdiff(R, B)

            # 分级通道：gray 命中 → 返回；不命中 → 试增强 → 试彩色
            stages = [
                [gray],
                [enhanced, cv2.bitwise_not(enhanced)],
                [sat, cv2.bitwise_not(sat), rb_diff, cv2.bitwise_not(rb_diff)],
            ]
            best_v: Optional[int] = None
            best_conf = -1.0
            for stage in stages:
                for img in stage:
                    try:
                        r, _ = reader(img)
                    except Exception:
                        continue
                    if not r:
                        continue
                    for item in r:
                        if not isinstance(item, (list, tuple)) or len(item) < 3:
                            continue
                        text = str(item[1]).strip().rstrip(".。,")
                        conf = float(item[2])
                        if not text.isdigit() or conf < 0.25:
                            continue
                        v = int(text)
                        if 1 <= v <= 99 and conf > best_conf:
                            best_v = v
                            best_conf = conf
                if best_v is not None:
                    break  # 当前 stage 命中即退出，节省后续通道
            results.append(best_v)
        return results

    # ------------------------------------------------------------------
    def _ocr_per_die_via_bailian(self, img_bgr: np.ndarray, bboxes: List[List[int]]) -> List[int]:
        """每颗骰子单独裁剪后调百炼 OCR。最准（focus 在单颗上），最慢（N 次 API）。"""
        import subprocess, tempfile, re
        if not bboxes:
            return []
        results: List[int] = []
        H, W = img_bgr.shape[:2]
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            pad = 12
            xa = max(0, x1 - pad); ya = max(0, y1 - pad)
            xb = min(W, x2 + pad); yb = min(H, y2 + pad)
            crop = img_bgr[ya:yb, xa:xb]
            # 放大提升小字准确率
            ch, cw = crop.shape[:2]
            scale = max(1.5, 240 / max(ch, cw))
            crop_big = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                cv2.imwrite(tmp.name, crop_big)
                proc = subprocess.run(
                    ["bl", "vision", "describe",
                     "--image", tmp.name,
                     "--prompt", "图中最显眼的数字（骰子顶面），只输出那个数字本身，不要任何解释。",
                     "--model", "qwen-vl-ocr"],
                    capture_output=True, text=True, timeout=30,
                )
                text = (proc.stdout or "").strip()
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            v = None
            for chunk in re.split(r"[^\d]+", text):
                if chunk.isdigit():
                    n = int(chunk)
                    if 1 <= n <= 99:
                        v = n
                        break
            if v is not None:
                results.append(v)
        return results

    # ------------------------------------------------------------------
    def _detect_dice_polyhedral(self, img_bgr: np.ndarray, fast: bool = False) -> tuple[np.ndarray, List[List[int]]]:
        """检测多面骰（D4~D24）bbox，返回 [x1,y1,x2,y2] 列表。
        fast=True：极简路径，专为摄像头实时（~10-30ms/帧）
        fast=False：完整多模态融合 + 凸性/纹理校验（~50-100ms/帧）
        """
        annotated = img_bgr.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if fast:
            # 极速路径：自适应阈值 + 颜色距离局部均值 + 最小形态学（覆盖同色场景）
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            m1 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 3,
            )
            local_mean = cv2.GaussianBlur(img_bgr, (41, 41), 0)
            color_diff = cv2.absdiff(img_bgr, local_mean)
            diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
            _, m2 = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_or(m1, m2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = (w * h) * 0.003
            max_area = (w * h) * 0.25
            bboxes: List[List[int]] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 35 or bh < 35:
                    continue
                ratio = bw / max(bh, 1)
                if ratio < 0.4 or ratio > 2.5:
                    continue
                bboxes.append([int(x), int(y), int(x + bw), int(y + bh)])
            return annotated, bboxes

        # 完整路径：多模态分割 + 严格校验
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        m1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5,
        )
        local_mean = cv2.GaussianBlur(img_bgr, (61, 61), 0)
        color_diff = cv2.absdiff(img_bgr, local_mean)
        diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        _, m2 = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(blurred, 30, 90)
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        m3 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_big, iterations=3)

        combined = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
        kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_med, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_med, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (w * h) * 0.004
        max_area = (w * h) * 0.22
        bboxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 40 or bh < 40:
                continue
            ratio = bw / max(bh, 1)
            if ratio < 0.45 or ratio > 2.2:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.55:
                    continue
            roi_edges = cv2.Canny(blurred[y:y + bh, x:x + bw], 30, 90)
            edge_density = roi_edges.sum() / 255.0 / (bw * bh)
            if edge_density < 0.005:
                continue
            bboxes.append([int(x), int(y), int(x + bw), int(y + bh)])

        return annotated, bboxes