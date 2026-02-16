"""
MAE pretraining loop.
Logs reconstruction loss, saves checkpoints.
"""
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any

from ..models import MAE
from ..datasets import get_dtd_loaders
from ..utils import set_seed
from ..utils.config import load_config
from .trainer import get_optimizer, get_cosine_schedule, save_checkpoint
from ..utils.logging import get_log_dir, get_ckpt_dir


def build_mae(cfg: dict[str, Any], device: torch.device) -> MAE:
    m = cfg.get("model", {})
    d = cfg.get("data", {})
    img_size = d.get("image_size", 224)
    model = MAE(
        img_size=img_size,
        patch_size=m.get("patch_size", 16),
        in_chans=m.get("in_chans", 3),
        embed_dim=m.get("embed_dim", 192),
        depth=m.get("depth", 12),
        num_heads=m.get("num_heads", 3),
        mlp_ratio=m.get("mlp_ratio", 4.0),
        decoder_embed_dim=m.get("decoder_embed_dim", 512),
        decoder_depth=m.get("decoder_depth", 8),
        decoder_num_heads=m.get("decoder_num_heads", 16),
        norm_pix_loss=m.get("norm_pix_loss", True),
    )
    return model.to(device)


def train_mae(config_path: str, device: torch.device | None = None) -> Path:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_loader, _, _ = get_dtd_loaders(
        root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        image_size=data_cfg["image_size"],
        split_index=data_cfg.get("split", 1),
    )

    model = build_mae(cfg, device)
    mask_ratio = cfg.get("mask_ratio", 0.75)
    run_name = cfg.get("run_name", "mae")
    paths = cfg.get("paths", {})
    log_dir = get_log_dir(paths.get("log_dir", "experiments/logs"), run_name)
    ckpt_dir = get_ckpt_dir(paths.get("ckpt_dir", "experiments/checkpoints"), run_name)

    train_cfg = cfg.get("training", {})
    epochs = int(train_cfg.get("epochs", 150))
    lr = float(train_cfg.get("lr", 1.5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 10))
    min_lr = float(train_cfg.get("min_lr", 1e-6))

    optimizer = get_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    scheduler = get_cosine_schedule(optimizer, total_steps, warmup_steps=warmup_steps, min_lr_ratio=min_lr / lr)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, step=epoch * steps_per_epoch + step, epoch=epoch, extra={"mask_ratio": mask_ratio})
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step=(epoch + 1) * steps_per_epoch - 1, epoch=epoch, extra={"mask_ratio": mask_ratio})
    return ckpt_dir
