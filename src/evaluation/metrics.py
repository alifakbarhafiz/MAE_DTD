"""PSNR and SSIM for reconstruction quality."""
import torch
import numpy as np
from math import log10


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    pred, target: (B, C, H, W) in [0, 1] or normalized.
    Returns mean PSNR in dB.
    """
    mse = (pred - target).pow(2).mean().item()
    if mse <= 0:
        return float("inf")
    return 10 * log10(max_val ** 2 / mse)


def _to_numpy_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    if x.max() <= 1.0 and x.min() >= 0:
        x = (x * 255).clamp(0, 255)
    return x.numpy().astype(np.uint8)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> float:
    """
    Simplified SSIM: single scale, no window (or use skimage/cv2).
    pred, target: (B, C, H, W).
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        return 0.0
    pred_np = _to_numpy_uint8(pred)
    target_np = _to_numpy_uint8(target)
    if pred_np.ndim == 3:
        pred_np = pred_np[np.newaxis]
        target_np = target_np[np.newaxis]
    vals = []
    for i in range(pred_np.shape[0]):
        s = ssim_fn(
            pred_np[i].transpose(1, 2, 0),
            target_np[i].transpose(1, 2, 0),
            multichannel=True,
            channel_axis=2,
            data_range=255,
        )
        vals.append(s)
    return float(np.mean(vals)) if size_average else vals
