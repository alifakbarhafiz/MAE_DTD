"""
Visualize reconstructions and save comparison grids.
Optionally compute PSNR/SSIM.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Any

from ..models.mae import MAE, patchify, unpatchify
from ..datasets import get_dtd_loaders
from ..utils.config import load_config
from ..training.trainer import load_checkpoint
from .metrics import compute_psnr, compute_ssim

# ImageNet normalization for denormalizing
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) normalized -> (B, C, H, W) [0,1]."""
    t = t.detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def build_mae_from_config(cfg: dict[str, Any], device: torch.device) -> MAE:
    from ..training.train_mae import build_mae
    return build_mae(cfg, device)


def run_reconstruction(
    config_path: str,
    checkpoint_path: str,
    output_dir: str | Path,
    device: torch.device | None = None,
    num_samples: int = 16,
    mask_ratio: float | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)
    mask_ratio = mask_ratio or cfg.get("mask_ratio", 0.75)

    _, _, test_loader = get_dtd_loaders(
        root=cfg["data"]["root"],
        batch_size=min(num_samples, cfg["data"]["batch_size"]),
        num_workers=0,
        image_size=cfg["data"]["image_size"],
        split_index=cfg["data"].get("split", 1),
    )

    model = build_mae_from_config(cfg, device)
    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(test_loader)
    imgs, _ = next(data_iter)
    imgs = imgs.to(device)[:num_samples]

    with torch.no_grad():
        _, pred_patches, mask = model(imgs, mask_ratio=mask_ratio)

    patch_size = model.patch_size
    C = model.in_chans
    if model.norm_pix_loss:
        target = patchify(imgs, patch_size)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True) + 1e-6
        pred_denorm = pred_patches * var.sqrt() + mean
    else:
        pred_denorm = pred_patches
    rec = unpatchify(pred_denorm, patch_size, C)
    rec = denormalize(rec)
    orig = denormalize(imgs)

    psnr = compute_psnr(rec, orig)
    try:
        ssim = compute_ssim(rec, orig)
    except Exception:
        ssim = 0.0

    try:
        import matplotlib.pyplot as plt
        n = min(4, num_samples)
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
        for i in range(n):
            axes[0, i].imshow(orig[i].permute(1, 2, 0).numpy())
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")
            axes[1, i].imshow(rec[i].permute(1, 2, 0).numpy())
            axes[1, i].set_title("Recon")
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "reconstruction_grid.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    return {"psnr": psnr, "ssim": ssim}
