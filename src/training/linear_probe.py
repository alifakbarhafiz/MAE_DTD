"""
Linear probe: freeze pretrained encoder, train linear classifier on 47 DTD classes.
"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Any

from ..models import MAE
from ..datasets import get_dtd_loaders
from ..utils import set_seed
from ..utils.config import load_config
from .trainer import load_checkpoint

# DTD number of classes
NUM_CLASSES = 47


class LinearProbe(nn.Module):
    """Freeze encoder, train linear head on mean-pooled patch tokens."""

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x, return_cls=False)
        z = z.mean(dim=1)
        return self.head(z)


def _get_encoder_from_mae(mae: MAE) -> nn.Module:
    return mae.encoder


def run_linear_probe(
    config_path: str,
    checkpoint_path: str,
    device: torch.device | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader = get_dtd_loaders(
        root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        image_size=data_cfg["image_size"],
        split_index=data_cfg.get("split", 1),
    )

    m_cfg = cfg.get("model", {})
    embed_dim = m_cfg.get("embed_dim", 192)
    mae = MAE(
        img_size=data_cfg.get("image_size", 224),
        patch_size=m_cfg.get("patch_size", 16),
        in_chans=m_cfg.get("in_chans", 3),
        embed_dim=embed_dim,
        depth=m_cfg.get("depth", 12),
        num_heads=m_cfg.get("num_heads", 3),
        mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
        decoder_embed_dim=m_cfg.get("decoder_embed_dim", 512),
        decoder_depth=m_cfg.get("decoder_depth", 8),
        decoder_num_heads=m_cfg.get("decoder_num_heads", 16),
        norm_pix_loss=m_cfg.get("norm_pix_loss", True),
    )
    load_checkpoint(Path(checkpoint_path), mae, device=device)
    encoder = _get_encoder_from_mae(mae)
    for p in encoder.parameters():
        p.requires_grad = False

    model = LinearProbe(encoder, embed_dim, NUM_CLASSES).to(device)
    # Linear probe needs to use encoder's patch_embed; encoder is ViT with patch_embed
    # But LinearProbe uses encoder.patch_embed - and MAE's encoder has patch_embed. Good.
    # For representation we need full image forward: patch_embed then mean pool (no masking for probe)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Probe epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy
