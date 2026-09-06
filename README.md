# YOLO Web Studio

本地网页版 YOLO 工作台。两个任务：

1. **🎲 跑团骰子点数识别**：上传一张多颗骰子的照片，框出每颗并读出点数，输出总和。
2. **📖 文档书籍版面拉框**：上传一本书/文档的页面，框出标题/段落/图/表/页眉等区域（不做 OCR）。

## 技术栈

- Web：Flask（本地 1111 端口）
- 模型：Ultralytics YOLOv8（用魔搭 ModelScope 拉权重到本地）
  - 文档版面：`opendatalab/DocLayout-YOLO-DocStructBench` (魔搭镜像)
- 骰子检测：传统 OpenCV（自适应阈值 + 形态学闭运算 + 轮廓外接矩形，多边形 3+ 边覆盖 D4~D24）
- 骰子点数识别：阿里云百炼 `qwen-vl-ocr`（一次调用读全图所有骰子顶面数字）
- 推理后端：GPU（GTX 1650，4GB），自动 fallback CPU

## 目录结构

```
dev/yolo/
├── app.py                  # Flask 主程序
├── models/
│   ├── doclayout_yolo.pt   # 文档版面 YOLO 权重（魔搭拉）
│   └── dice_yolo.pt        # 骰子检测权重（魔搭/Ultralytics 拉）
├── static/
│   ├── style.css
│   └── app.js
├── templates/
│   └── index.html
├── uploads/                # 用户上传
├── results/                # 推理结果图
└── requirements.txt
```

## 启动

```bash
# 装依赖
pip install -r requirements.txt
# 配置百炼 API key（OCR 推荐）
export DASHSCOPE_API_KEY=sk-xxxxxxxx

# 用 run.sh 启动（自动注入 CUDA 12 库到 LD_LIBRARY_PATH，让 onnxruntime-gpu 工作）
./run.sh            # HTTP
./run.sh --https    # HTTPS（推荐，局域网摄像头跨设备访问）
# 浏览器打开 http(s)://localhost:1111  或 http(s)://<本机IP>:1111
```

GPU：GTX 1650 (CUDA 12.4) 会自动被 torch + onnxruntime-gpu 使用。
骰子识别：
- 上传模式默认走百炼 `qwen-vl-ocr` per-die（最准，~3-7s）
- 摄像头模式默认走本地 RapidOCR per-die（GPU ~2-4s / CPU ~4-7s）
- 摄像头模式勾选「走百炼」→ 本地先检出 bbox，有 bbox 才 per-die API（省配额）

`--https` 用自签名证书，浏览器首次访问提示「不安全」→ 点「高级 → 继续前往」即可。

## 模型下载（首次运行）

模型权重自动从魔搭 ModelScope 拉到 `models/`，不联网时不会断网去 HF。# DND_dice
