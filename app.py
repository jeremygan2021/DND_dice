"""
YOLO Web Studio — Flask 主程序
两个任务：
  - dice      : 跑团骰子检测 + 点数识别（裁剪 + HoughCircles）
  - doclayout : 文档版面拉框（不做 OCR）
模型从魔搭 ModelScope 拉到本地 models/，不依赖 HF。
"""
from __future__ import annotations

import ctypes
import io
import os
import time
import uuid
import logging
from pathlib import Path


def _preload_nvidia_libs():
    """onnxruntime-gpu 通过 dlopen 找 CUDA 库，先预加载避免找不到。"""
    base = Path(__file__).resolve().parent / "venv/lib/python3.13/site-packages/nvidia"
    paths = [
        ("cuda_runtime/lib", "libcudart.so.12"),
        ("cuda_nvrtc/lib", "libnvrtc.so.12"),
        ("cublas/lib", "libcublas.so.12"),
        ("cublas/lib", "libcublasLt.so.12"),
        ("cuda_cupti/lib", "libcupti.so.12"),
        ("cudnn/lib", "libcudnn.so.9"),
        ("curand/lib", "libcurand.so.10"),
        ("cufft/lib", "libcufft.so.11"),
        ("cusolver/lib", "libcusolver.so.11"),
        ("cusparse/lib", "libcusparse.so.12"),
        ("nccl/lib", "libnccl.so.2"),
    ]
    for sub, lib in paths:
        full = base / sub / lib
        if full.exists():
            try:
                ctypes.CDLL(str(full))
            except OSError:
                pass


_preload_nvidia_libs()

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

from model_loader import ModelLoader

# ----------------------------------------------------------------------
# 路径与日志
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("yolo-web")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# 单例模型加载器（首次请求才真正下载/加载）
loader = ModelLoader(BASE_DIR / "models")


# ----------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "models": loader.status(),
        "device": loader.device,
    })


@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname)


@app.route("/results/<path:fname>")
def serve_result(fname):
    return send_from_directory(RESULT_DIR, fname)


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """统一推理入口。表单字段：task=dice|doclayout, image=<file>, mode=upload|camera"""
    task = request.form.get("task", "").strip().lower()
    if task not in ("dice", "doclayout"):
        return jsonify({"error": "task 必须为 dice 或 doclayout"}), 400

    mode = request.form.get("mode", "upload").strip().lower()
    if mode not in ("upload", "camera"):
        return jsonify({"error": "mode 必须为 upload 或 camera"}), 400

    if "image" not in request.files:
        return jsonify({"error": "缺少 image 文件"}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "空文件"}), 400

    raw = f.read()
    np_arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "图片读取失败"}), 400
    h, w = img_bgr.shape[:2]

    t0 = time.time()
    try:
        if task == "doclayout":
            annotated, blocks = loader.infer_doclayout(img_bgr)
            summary = {
                "image_size": [w, h],
                "num_blocks": len(blocks),
                "blocks": blocks,
            }
        else:  # dice
            api_requested = (mode == "camera") and (request.form.get("api", "0") == "1")
            use_fast_cv = (mode == "camera")  # 摄像头模式用极速 CV 路径

            if mode == "upload":
                # 上传：完整 CV + per-die API（最准）
                annotated, dice_info = loader.infer_dice(img_bgr, use_api=True, per_die_api=True, fast=False)
            elif api_requested:
                # 摄像头 + API：极速 CV 找 bbox，无 bbox 不调 API
                _ann, bboxes = loader._detect_dice_polyhedral(img_bgr, fast=True)
                bboxes.sort(key=lambda b: (b[1], b[0]))
                if not bboxes:
                    dice_info = []
                    annotated = img_bgr.copy()
                else:
                    api_values = loader._ocr_per_die_via_bailian(img_bgr, bboxes)
                    dice_info = [
                        {"bbox": b, "value": int(v), "source": "api"}
                        for b, v in zip(bboxes, api_values)
                    ]
                    annotated = img_bgr.copy()
                    total = sum(d["value"] for d in dice_info)
                    header = f"骰子数: {len(dice_info)}   总点数: {total}   (API校准)"
                    cv2.rectangle(annotated, (0, 0), (560, 36), (0, 0, 0), -1)
                    cv2.putText(annotated, header, (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    for d in dice_info:
                        x1, y1, x2, y2 = d["bbox"]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)
                        (tw, th), _ = cv2.getTextSize(str(d["value"]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 100), -1)
                        cv2.putText(annotated, str(d["value"]), (x1 + 3, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                # 摄像头 + 本地 OCR：极速 CV + per-die 多通道 OCR
                annotated, dice_info = loader.infer_dice(img_bgr, use_api=False, per_die_api=False, fast=True)
            total = sum(d["value"] for d in dice_info)
            summary = {
                "image_size": [w, h],
                "num_dice": len(dice_info),
                "total_value": total,
                "dice": dice_info,
            }
    except Exception as e:
        log.exception("推理失败")
        return jsonify({"error": f"推理失败: {e}"}), 500

    elapsed_ms = int((time.time() - t0) * 1000)

    # 摄像头模式：不落盘，直接返回 bbox 让前端自己叠框
    if mode == "camera":
        return jsonify({
            "task": task,
            "mode": "camera",
            "elapsed_ms": elapsed_ms,
            "image_size": [w, h],
            "dice": summary["dice"] if task == "dice" else [],
            "total_value": summary.get("total_value", 0),
            "num_dice": summary.get("num_dice", 0),
        })

    # 上传模式：保存到磁盘，返回 URL
    uid = uuid.uuid4().hex[:12]
    ext = Path(f.filename).suffix.lower() or ".jpg"
    upload_name = f"{uid}{ext}"
    upload_path = UPLOAD_DIR / upload_name
    upload_path.write_bytes(raw)

    if task == "doclayout":
        result_name = f"doclayout_{uid}.jpg"
    else:
        result_name = f"dice_{uid}.jpg"
    result_path = RESULT_DIR / result_name
    cv2.imwrite(str(result_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return jsonify({
        "task": task,
        "mode": "upload",
        "elapsed_ms": elapsed_ms,
        "result_url": f"/results/{result_name}",
        "upload_url": f"/uploads/{upload_name}",
        "summary": summary,
    })


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _make_adhoc_ssl():
    """生成自签名证书（adhoc SSL），用于局域网 HTTPS 摄像头访问。"""
    from OpenSSL import crypto
    import socket

    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)

    cert = crypto.X509()
    cert.get_subject().C = "CN"
    cert.get_subject().ST = "Local"
    cert.get_subject().L = "LAN"
    cert.get_subject().O = "YOLO Web Studio"
    cert.get_subject().OU = "dev"
    cert.get_subject().CN = "yolo.local"
    cert.get_subject().emailAddress = "dev@local"
    cert.set_serial_number(1)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(60 * 60 * 24 * 365)  # 1 年
    cert.set_issuer(cert.get_subject())

    # SAN 扩展：本机所有 IP
    sans = [b"DNS:yolo.local", b"DNS:localhost", b"IP:127.0.0.1"]
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            sans.append(f"IP:{ip}".encode())
    except Exception:
        pass
    sans.append(b"IP:0.0.0.0")
    from OpenSSL.crypto import X509Extension
    cert.add_extensions([
        X509Extension(
            b"subjectAltName",
            critical=False,
            value=b", ".join(sans),
        ),
        X509Extension(b"basicConstraints", critical=True, value=b"CA:FALSE"),
    ])

    cert.set_pubkey(key)
    cert.sign(key, "sha256")
    return (crypto.dump_certificate(crypto.FILETYPE_PEM, cert),
            crypto.dump_privatekey(crypto.FILETYPE_PEM, key))


if __name__ == "__main__":
    import sys
    use_https = "--https" in sys.argv
    if use_https:
        log.info("启动 HTTPS（自签名证书）...")
        cert_pem, key_pem = _make_adhoc_ssl()
        # 落盘到 tmp（Flask adhoc 需要文件路径）
        import tempfile, os
        crt = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
        key = tempfile.NamedTemporaryFile(delete=False, suffix=".key")
        crt.write(cert_pem); crt.close()
        key.write(key_pem); key.close()
        ssl_ctx = (crt.name, key.name)
        log.info("YOLO Web Studio 启动 → https://0.0.0.0:1111")
        log.info("首次访问浏览器会提示证书不受信任，点「高级 → 继续访问」即可")
        app.run(host="0.0.0.0", port=1111, debug=False, threaded=True, ssl_context=ssl_ctx)
    else:
        log.info("YOLO Web Studio 启动 → http://0.0.0.0:1111")
        log.info("如需局域网 HTTPS（摄像头跨设备访问），加 --https 参数启动")
        app.run(host="0.0.0.0", port=1111, debug=False, threaded=True)