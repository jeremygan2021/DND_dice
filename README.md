# 跑团骰子识别工作台

Flask 网页支持上传图片和摄像头识别，同时保留文档版面检测功能。

## 当前实现与边界

- **专用 YOLO + OCR**：推荐用于混合 D4/D6/D8/D10/D12/D20/D100。加载本地 `models/dice.pt`，七个类别名必须为 `d4,d6,d8,d10,d12,d20,d100`（顺序不限）。通用 COCO 权重、文档模型不能用于骰型分类。
- **CV 回退**：没有权重时自动使用颜色/轮廓检测、凸性过滤和重复框抑制。适合纯色骰盘，无法可靠区分骰型。界面显示 `unknown`，同种骰子可手动选类型来限制 OCR 数值范围。混合骰必须选择自动。
- **OCR**：本地 RapidOCR，彩色/增强候选及 15° 步进旋转，一致性投票和骰型值域检查。多个可见数字冲突时返回 `null`，前端显示“待确认”；这是保守读数策略，仍不能保证找到物理顶面。
- **D4**：按顶角重复数字布局处理，要求裁剪中心附近至少两处数字一致，再跨候选验证。底边布局不适用；遮挡、视角不佳时可能拒识。该规则尚未在真实 D4 标注集上验证。**VLM 兜底**：百炼 VLM 在该规则失败时按"三面顶角数字之和 = 顶面"提示重读；qwen3-vl-flash/plus 在多数实拍下能稳定给出 D4 顶面，但仍需更多样本验证。
- **D10**：已知类型时 `0 → 10`；未知类型的单个 `0` 不猜测。
- **D100**：用户指定的百分位十面骰，仅接受 `00,10,…,90`，`00 → 0`。界面的总点数是已确认读数的算术和，**不自动将 D100 与 D10 配对**（例如双零掷出 100 的组合规则需要单独处理）。RapidOCR 的检测网络常把"70"等两位拆成两个独立单字框，本地启发式会尝试按位置/高度把它们拼回去；若拼不上，**VLM 兜底**会按专用两位数 prompt 直接让云端读两位，本地仅做值域过滤。位置规则：当画面中混合 D10/D100 且用户未手动选型时，按"少颗 = D100、多颗 = D10"自动归类（前提是 D100 单独放一侧、形成明显间隔）。手动指定 D100 后位置规则不生效。
- **摄像头**：过滤短暂出现的框；稳定后后台 OCR；移动/换面清除旧值；每 0.5 秒至少重检测一次以发现进入/离开的骰子。OCR 队列最多 16 个任务，异常可退避重试。

仓库目前没有骰子训练权重或标注数据，七类自动识别还不能验收为完成。提供的两张蓝色实拍照片目前各检出两框，读数分别为 4/4、6/7；CPU 首次 OCR 每张约 8～11 秒。这些照片参与了调试，不能作为独立准确率评估。套装商品图检出 7 框，但多面数字选择仍不可靠，例如 D20 会读到中央侧面的 8；因此不能将商品图读数作为掷骰结果。没有真实数据集精度报告，也没有 GPU 性能实测。

## 启动

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
./run.sh             # HTTP :1111
./start.sh --https   # 局域网摄像头，需信任自签名证书
```

摄像头通常要求 HTTPS 或 localhost。上传和摄像头都支持“画面内骰型”选择。

## 接入专用权重

```bash
DICE_MODEL=/absolute/path/best.pt DICE_DETECTOR=yolo ./run.sh
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DICE_DETECTOR` | `auto` | `auto` / `cv` / `yolo`；明确选 yolo 缺权重则报错 |
| `DICE_MODEL` | 项目 `models/dice.pt` | 本地七类检测模型 |
| `DICE_CONF` | `0.55` | YOLO 检测置信度 |
| `DICE_DEVICE` | CUDA 可用则 `cuda:0`，否则 `cpu` | YOLO 设备 |
| `CV_THREADS` | `2` | OpenCV CPU 线程数 |

YOLO 使用跨类别 NMS、包含框抑制、最多 30 框；CUDA 模式启用 FP16。模型实例推理串行，避免多会话同时访问 predictor。

## 数据和训练

准备 YOLO detection 格式：每张图片对应同名 `.txt`，每行 `class x_center y_center width height`，坐标归一化。标注整颗骰子，不能把每个数字或侧面当成独立骰子。加入空桌面、手、文字、阴影等负样本。训练集与验证集按拍摄场景/骰子套装分开，避免同一视频相邻帧泄漏。

数据配置示例（路径换成实际数据目录）：

```yaml
path: /absolute/path/dice-dataset
train: images/train
val: images/val
names:
  0: d4
  1: d6
  2: d8
  3: d10
  4: d12
  5: d20
  6: d100
```

```bash
venv/bin/python train_dice.py --data /path/dice.yaml --model /path/yolov8n.pt --device cuda:0
venv/bin/python train_dice.py --data /path/dice.yaml --model /path/best.pt --device cuda:0 --validate
```

脚本只接受现有本地模型文件，不隐式下载通用权重。训练输出在 `runs/dice/`，验收后将 best.pt 配置为 `DICE_MODEL`。除检测 mAP 外，还需人工标注上面读数，统计逐骰最终准确率、拒识率、误框数和延迟；YOLO 的 mAP 不能代表 OCR 正确率。

## GPU

`/healthz` 提供 PyTorch 实际 CUDA 探测、OpenCV CUDA 设备数、ONNX 已注册 provider，以及 OCR 初始化后的实际 session providers。未初始化 OCR 时 providers 为空。

当前 Python 3.13 环境只能安装旧分支 RapidOCR 1.2.3；`ocr_compat.py` 修正该版本 CTC 分数多平均一个零值的错误（单字分数原本被减半），在上游过滤前恢复原始平均置信度。Python 3.12 及以下使用 1.3.24+。未强行忽略包的 Python 版本限制。

当前机器 PyTorch CUDA 不可用，OpenCV CUDA 设备为 0，OCR 实际使用 CPU。仅安装 CUDA 版 Python 包不能替代 NVIDIA 驱动/显卡透传。部署 GPU 时应安装与 PyTorch、CUDA、cuDNN 兼容的 `onnxruntime-gpu`，同一环境不要同时保留 CPU 与 GPU 两个 ORT 包。代码通过运行时发现能力和可用的 ORT `preload_dlls()` 加载依赖，不再硬编码 Python 3.13 的库路径。

参考：[Ultralytics 推理参数](https://docs.ultralytics.com/modes/predict/)、[ONNX CUDA 依赖矩阵](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

## API 与检查

`POST /api/infer` multipart：`task=dice`、`mode=upload|camera`、`image`、可选 `dice_type`；摄像头重复传同一 `session`。
返回每颗骰子的 `bbox,value,text,conf,dice_type`；上传也保留未读出数字的框。

原有可选百炼开关仍保留，需自行配置 CLI/API 凭证；本次未调用外部 API，也未验证该路径的识别效果。D4 始终使用本地重复数字规则。
文档任务仍由 `model_loader.py` 从魔搭获取文档模型，不参与骰子识别。

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python -m compileall -q app.py dice_engine.py dice_detector.py hardware.py train_dice.py
node --check static/app.js
```

实拍复测（输出标注图和 JSON，不把模型输出当成正确标签）：

```bash
venv/bin/python evaluate_dice.py /path/photo1.jpg /path/photo2.jpg --output results/validation
```


## 本地 dieCamera ONNX 接入审计（2026-09-06）

已保留现有后端切换功能并修正以下问题：

- 下载的 shape/glyph `.classes.json` 标记为 YOLOX，输出分别为 `(1,8400,11)` / `(1,8400,10)`；按 objectness × class 解码，同时支持标准 YOLOv8 输出布局。
- `dice-value-cls` 已输出 60 路概率，不能再次 softmax。此 Ultralytics 分类器使用 RGB 0～1；`dice-value` ConvNeXt 才使用 ImageNet 均值/方差。二者预处理不可混用。
- glyph 恢复整帧推理，按骰型、中心和覆盖率配对到骰子，同一帧多个骰子共享缓存。不能把逐骰裁剪推理称为与整帧等价。
- 读数按全类别概率检查（默认最低 0.65，前两名差至少 0.15），不把错误骰型限定后的概率重新归一化来制造高置信度。两个有效读数模型冲突时不计总数；未命中时允许本地 OCR 兜底，响应 `source` 记录真实来源。
- 请求后端通过 ContextVar 隔离，后台任务捕获后端和原始帧，不再因为某个模型已加载就污染 CV 请求。切换后端会清除原跟踪缓存。
- 只在实际 CUDA 可用时申请 CUDA；健康检查显示每个 ONNX session 的实际 provider。CPU 线程默认 2，可通过 `DICE_ONNX_THREADS` 调整。

**模型边界：**该套模型只包含 d4/d6/d8/d10/d12/d20，不能自动区分 D100。手动指定 D100 时改用数字 OCR，按 00～90 限定；混合画面仍需百分位专用标注和再训练。`dice-face-geom.onnx` 只有 5 路原始变换输出，没有已验证的变换/预处理说明，未盲目接入。

修正后两张旧实拍图共检测 4 颗，读出 4/4/6，余下一颗因为模型判为 d6 而实际可见 7 被拒识；这 3 个数均来自本地 OCR 回退，不是 ONNX 分类器已达到 75% 泛化准确率。CPU 分别约 14.3 秒、11.2 秒，仍需优化/训练。结果见 `results/onnx-audit/after/report.json`。现有截图带界面数字和框，不作为读数测试集。

摄像头新增 720p/1080p/1440p 取帧选项，默认申请 1080p（设备可能回退），JPEG 质量 0.92。提高分辨率会增加传输成本；缩小实际拍摄范围、使用均匀散射光以减少镜面反射，同样关键。插值放大不能恢复原本缺失的笔画。

下一步训练应标注：整骰 bbox + 骰型、向上读数的数字区域 + 正确值，以及 D4 顶角规则、D100 百分位标签。按不同骰子套装/拍摄场景分开训练和验证，保留未参与调参的真实测试集。当前 ONNX 是推理导出文件，不能直接传给 Ultralytics 做继续训练；需原始训练 checkpoint，或从可训练的基础权重重新微调。
