#!/usr/bin/env bash
# yolo-web 启动包装：把 nvidia CUDA 12 lib 路径注入 LD_LIBRARY_PATH
set -e
NV_BASE="/home/kali/dev/yolo/venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV_BASE/cuda_runtime/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cudnn/lib:$NV_BASE/curand/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$NV_BASE/nccl/lib:$LD_LIBRARY_PATH"
exec /home/kali/dev/yolo/venv/bin/python /home/kali/dev/yolo/app.py "$@"
