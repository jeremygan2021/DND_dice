"""Runtime capabilities; registered providers do not prove working GPU execution."""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def capabilities():
    result = {'cuda_available': False, 'device': 'cpu', 'gpu_name': None,
              'onnx_available_providers': [], 'opencv_cuda_devices': 0}
    try:
        import torch
        if torch.cuda.is_available():
            torch.empty(1, device='cuda').add_(1)
            result.update(cuda_available=True, device='cuda:0', gpu_name=torch.cuda.get_device_name(0))
    except Exception as exc:
        result['cuda_error'] = str(exc)
    try:
        import onnxruntime as ort
        if result['cuda_available'] and hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        result['onnx_available_providers'] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import cv2
        cv2.setNumThreads(max(1, int(os.getenv('CV_THREADS', '2'))))
        result['opencv_cuda_devices'] = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        pass
    return result
