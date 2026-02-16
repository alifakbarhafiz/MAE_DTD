"""
Extract encoder embeddings, compute covariance, eigenvalue spectrum, effective rank.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Any

from ..models import MAE
from ..datasets import get_dtd_loaders
from ..utils.config import load_config
from ..training.trainer import load_checkpoint


def run_spectrum_analysis(
    config_path: str,
    checkpoint_path: str,
    output_dir: str | Path,
    device: torch.device | None = None,
    max_batches: int = 50,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)

    _, _, test_loader = get_dtd_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
        image_size=cfg["data"]["image_size"],
        split_index=cfg["data"].get("split", 1),
    )

    from ..training.train_mae import build_mae
    model = build_mae(cfg, device)
    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.eval()

    encoder = model.encoder
    embed_dim = model.embed_dim
    all_embeds = []

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            imgs = imgs.to(device)
            z = encoder(imgs, return_cls=False)
            all_embeds.append(z.cpu().numpy())
    X = np.concatenate(all_embeds, axis=0)
    X = X.reshape(-1, embed_dim)
    X = X - X.mean(axis=0)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    trace = eigenvalues.sum()
    entropy = -np.sum(eigenvalues / trace * np.log(eigenvalues / trace + 1e-10))
    effective_rank = np.exp(entropy)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "eigenvalues.npy", eigenvalues)
    np.save(output_dir / "effective_rank.npy", np.array([effective_rank]))

    return {
        "effective_rank": float(effective_rank),
        "max_eigenvalue": float(eigenvalues[0]),
        "min_eigenvalue": float(eigenvalues[-1]),
    }
