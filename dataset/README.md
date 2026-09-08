# 训练数据汇总

仓库内所有训练数据从三个来源汇总，按用途分桶：

## 来源

| 路径 | 来源 | 用途 |
|---|---|---|
| `diecamera-crops/` | HF `G-G-Games/diecamera-crops`（CC BY-NC 4.0 / CC BY 4.0）| 1237 张单骰裁剪，仅 d4/d6/d8/d10/d12/d20 |
| `diecamera-frames/` | HF `G-G-Games/diecamera-frames`（CC BY-SA 4.0）| 460 张整盘帧 + 7 类 bbox |
| `custom-dice/` | 自采集 | 7 类点位 + 帧检测骨架（需自填）|
| `diecamera-models/` | HF `G-G-Games/diecamera-models`（AGPL-3.0）| 推理用 ONNX，不是训练数据 |

## 准备产物

| 路径 | 用途 | 由谁生成 |
|---|---|---|
| `diecamera-classify/{all60,d20}/` | Ultralytics 分类布局（点位） | `prepare_diecamera_crops.py` |
| `diecamera-detect/{images,labels}/` + `data.yaml` | Ultralytics detect 布局（七类） | `prepare_diecamera_frames.py` |
| `custom-dice/crops/{train,val,test}/<type>_<value>/` | 自有裁剪的七类点位 | `init_dice_capture.py`（目录骨架）|
| `custom-dice/frames/{train,val,test}/` | 自有整盘帧 + YOLO 标签（占位） | `init_dice_capture.py`（目录骨架）|

## 训练入口

| 任务 | 数据 | 脚本 |
|---|---|---|
| 七类点位分类（来自 diecamera-crops） | `diecamera-classify/all60/` | `train_diecamera.py --task all60` |
| D20 点位分类 | `diecamera-classify/d20/` | `train_diecamera.py --task d20` |
| 七类整盘检测 | `diecamera-detect/data.yaml` | `train_dice.py --data diecamera-detect/data.yaml --model yolo11n.pt` |
| 自有七类点位 | `custom-dice/crops/` | `train_diecamera.py --task custom7 --data-root dataset/custom-dice` |

D100 在 `diecamera-crops` 里没有出现（README 标注"one d100 die in the whole corpus"），
只能由 `custom-dice` 提供，或自行拍摄后追加到 `diecamera-crops` 重新跑
`prepare_diecamera_crops.py`。

## 重新下载数据集

```bash
venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('G-G-Games/diecamera-crops', repo_type='dataset',
                  local_dir='dataset/diecamera-crops')
snapshot_download('G-G-Games/diecamera-frames', repo_type='dataset',
                  local_dir='dataset/diecamera-frames')
"
```
