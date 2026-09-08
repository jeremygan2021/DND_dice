"""
YOLO Web Studio — Flask 主程序
================================
两个任务:
  - dice      : 跑团骰子检测 + 点数识别(两阶段引擎,见 dice_engine.py)
                · 摄像头模式:服务端会话跟踪,每帧只做「快速检测拉框」,
                  OCR 仅在该骰子稳定(连续多帧位移小 + ROI 内容不变)后
                  于后台触发一次并缓存;骰子再动/换面立即失效。
                · 上传模式:完整检测 + 逐骰高质量 OCR。
  - doclayout : 文档版面拉框(不做 OCR)
模型从魔搭 ModelScope 拉到本地 models/,不依赖 HF。
"""
from __future__ import annotations

import ctypes
import os
import time
import uuid
import logging
from pathlib import Path


from hardware import capabilities

capabilities()

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from model_loader import ModelLoader
import dice_engine
from dice_engine import registry as dice_registry

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

# 单例模型加载器(首次请求才真正下载/加载)
loader = ModelLoader(BASE_DIR / "models")

# 请求计数(周期性清理空闲摄像头会话)
_cleanup_counter = 0


# ----------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    onnx_status = {"available": False, "device": None}
    try:
        from dice_onnx_engine import engine as _onnx
        onnx_status["available"] = _onnx.available
        if _onnx._loaded:
            onnx_status["device"] = _onnx.device
            onnx_status["sessions"] = sorted(_onnx.sessions.keys())
            onnx_status["providers"] = {k:s.get_providers() for k,s in _onnx.sessions.items()}
    except Exception as e:
        onnx_status["error"] = str(e)
    return jsonify({
        "status": "ok",
        "models": loader.status(),
        "device": loader.device,
        "hardware": capabilities(),
        "dice_detector": dice_engine.detector.status(),
        "onnx_dice": onnx_status,
        "ocr_providers": dice_registry._ocr.providers,
    })


@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname)


@app.route("/results/<path:fname>")
def serve_result(fname):
    return send_from_directory(RESULT_DIR, fname)


def _decode_image(raw: bytes):
    np_arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, "图片读取失败"
    return img_bgr, None


def _camera_response(img_bgr, res: dict, elapsed_ms: int):
    h, w = img_bgr.shape[:2]
    return jsonify({
        "task": "dice",
        "mode": "camera",
        "elapsed_ms": elapsed_ms,
        "det_ms": res["det_ms"],
        "image_size": [w, h],
        "num_dice": res["num_dice"],
        "n_recognized": res["n_recognized"],
        "total_value": res["total_value"],
        "all_settled": res["all_settled"],
        "frame": res["frame"],
        "dice": res["dice"],
    })


def _handle_dice_camera(img_bgr, session_id: str, use_api: bool, dice_type="unknown",
                       vlm: bool = False, vlm_model: str = "qwen3-vl-flash"):
    global _cleanup_counter
    session = dice_registry.get(session_id)
    res = session.process_frame(img_bgr, dice_type, use_api, vlm=vlm, vlm_model=vlm_model)
    # 每 60 个请求顺手清理一次空闲会话
    _cleanup_counter += 1
    if _cleanup_counter % 60 == 0:
        dice_registry.cleanup()
    return res


class _DetectorModeCtx:
    """上下文管理器:临时把全局 DiceDetector 切到指定 backend,退出时还原。"""

    def __init__(self, mode: str):
        self.mode = mode
        self._prev = None

    def __enter__(self):
        from dice_detector import request_detector_mode
        self._token = request_detector_mode.set(self.mode if self.mode != "auto" else None)
        return self

    def __exit__(self, exc_type, exc, tb):
        from dice_detector import request_detector_mode
        request_detector_mode.reset(self._token)
        return False


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """统一推理入口。表单字段:task=dice|doclayout, image=<file>, mode=upload|camera"""
    task = request.form.get("task", "").strip().lower()
    if task not in ("dice", "doclayout"):
        return jsonify({"error": "task 必须为 dice 或 doclayout"}), 400

    dice_type = request.form.get("dice_type", "unknown").lower()
    if dice_type not in ("unknown", "d4", "d6", "d8", "d10", "d12", "d20", "d100"):
        return jsonify({"error": "无效骰型"}), 400

    mode = request.form.get("mode", "upload").strip().lower()
    if mode not in ("upload", "camera"):
        return jsonify({"error": "mode 必须为 upload 或 camera"}), 400

    detector_mode = request.form.get("detector", "auto").strip().lower()
    if detector_mode not in ("auto", "onnx", "cv", "yolo"):
        return jsonify({"error": "detector 必须为 auto/onnx/cv/yolo"}), 400

    vlm_enabled = request.form.get("vlm", "0") == "1"
    vlm_model = request.form.get("vlm_model", "qwen3-vl-flash").strip()
    VLM_ALLOWED = {"qwen3-vl-flash", "qwen3-vl-plus", "qwen-vl-ocr", "qwen3-vl-max"}
    if vlm_model not in VLM_ALLOWED:
        vlm_model = "qwen3-vl-flash"

    if "image" not in request.files:
        return jsonify({"error": "缺少 image 文件"}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "空文件"}), 400

    raw = f.read()
    img_bgr, err = _decode_image(raw)
    if img_bgr is None:
        return jsonify({"error": err}), 400
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
        elif mode == "camera":
            # —— 摄像头:会话化两阶段引擎 ——
            session_id = request.form.get("session", "").strip()
            if not session_id or len(session_id) > 64:
                session_id = "anon-" + uuid.uuid4().hex[:8]
            use_api = request.form.get("api", "0") == "1"
            with _DetectorModeCtx(detector_mode):
                res = _handle_dice_camera(img_bgr, session_id, use_api, dice_type,
                                          vlm=vlm_enabled, vlm_model=vlm_model)
            elapsed_ms = int((time.time() - t0) * 1000)
            return _camera_response(img_bgr, res, elapsed_ms)
        else:
            # —— 上传:完整检测 + 逐骰 OCR(可走百炼)——
            use_api = request.form.get("api", "0") == "1"
            with _DetectorModeCtx(detector_mode):
                annotated, dice_info = dice_engine.analyze_image(
                    img_bgr, use_api=use_api, per_die_api=use_api,
                    dice_type=dice_type,
                    vlm=vlm_enabled, vlm_model=vlm_model)
            summary = {
                "image_size": [w, h],
                "num_dice": len(dice_info),
                "total_value": sum(d["value"] for d in dice_info if d["value"] is not None),
                "dice": dice_info,
            }
    except Exception as e:
        log.exception("推理失败")
        return jsonify({"error": f"推理失败: {e}"}), 500

    elapsed_ms = int((time.time() - t0) * 1000)

    # 上传模式:保存到磁盘,返回 URL
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
    """生成自签名证书(adhoc SSL),用于局域网 HTTPS 摄像头访问。"""
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

    # SAN 扩展:本机所有 IP
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
        log.info("启动 HTTPS(自签名证书)...")
        cert_pem, key_pem = _make_adhoc_ssl()
        import tempfile
        crt = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
        key = tempfile.NamedTemporaryFile(delete=False, suffix=".key")
        crt.write(cert_pem); crt.close()
        key.write(key_pem); key.close()
        ssl_ctx = (crt.name, key.name)
        log.info("YOLO Web Studio 启动 → https://0.0.0.0:1111")
        log.info("首次访问浏览器会提示证书不受信任,点「高级 → 继续访问」即可")
        app.run(host="0.0.0.0", port=1111, debug=False, threaded=True, ssl_context=ssl_ctx)
    else:
        log.info("YOLO Web Studio 启动 → http://0.0.0.0:1111")
        log.info("如需局域网 HTTPS(摄像头跨设备访问),加 --https 参数启动")
        app.run(host="0.0.0.0", port=1111, debug=False, threaded=True)
