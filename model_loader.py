"""
model_loader.py — 模型加载与推理(现在只负责文档版面 doclayout)。

骰子检测 + OCR 已独立为 dice_engine.py(两阶段:实时拉框跟踪 / 稳定后 OCR)。
本模块仅保留:
  - doclayout YOLO 权重从魔搭 ModelScope 拉取 + GPU/CPU 推理
  - 设备探测与状态上报
"""
from __future__ import annotations

import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

log = logging.getLogger("yolo-web.loader")

# 模型魔搭 ID(用户指定:不要从 HF 拉,用魔搭 URL)
MODELSPEC_DOCLAYOUT = "opendatalab/DocLayout-YOLO-DocStructBench"


class ModelLoader:
    """单例懒加载器:首次用到某个模型才下载/初始化。"""

    # DocLayout-YOLO 类别(10 类,来自 DocStructBench)
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
        log.info(f"ModelLoader 初始化:device={self.device}, dir={model_dir}")

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
        return {"doclayout": self._doclayout is not None}

    # ==================================================================
    # 文档版面 YOLO
    # ==================================================================
    def _ensure_doclayout(self) -> Path:
        """从魔搭拉 doclayout 权重到本地。"""
        target = self.model_dir / "doclayout_yolo.pt"
        if target.exists() and target.stat().st_size > 1_000_000:
            log.info(f"doclayout 模型已存在:{target}")
            return target

        log.info("从魔搭 ModelScope 下载 DocLayout-YOLO 权重 ...")
        from modelscope import snapshot_download
        cache_dir = snapshot_download(
            MODELSPEC_DOCLAYOUT,
            cache_dir=str(self.model_dir.parent / ".ms_cache"),
        )
        pt_files = list(Path(cache_dir).rglob("*.pt"))
        if not pt_files:
            raise RuntimeError(f"魔搭包里没找到 .pt 权重:{cache_dir}")
        src = pt_files[0]
        shutil.copy2(src, target)
        log.info(f"doclayout 权重已落盘:{target} ({target.stat().st_size//1024//1024} MB)")
        return target

    def _load_doclayout(self):
        from ultralytics import YOLO
        weight = self._ensure_doclayout()
        log.info(f"加载 YOLO 模型:{weight}")
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
                cls_name = (self.DOCLAYOUT_CLASSES[cls_idx]
                            if cls_idx < len(self.DOCLAYOUT_CLASSES)
                            else f"cls_{cls_idx}")
                blocks.append({
                    "class": cls_name,
                    "conf": round(float(score), 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
        blocks.sort(key=lambda b: b["bbox"][1])
        return annotated, blocks
