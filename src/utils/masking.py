"""
Random patch masking for MAE.
Takes mask_ratio as input; returns visible indices (or mask).
"""

import torch


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Random patch masking.
    x: (B, N, C) patch tokens
    mask_ratio: fraction of patches to mask (0 to 1)
    Returns:
        - masked tokens (only visible): (B, N_vis, C)
        - mask: (B, N) True = visible, False = masked
        - ids_restore: (B, N) to restore full order from [vis | mask]
        - N_vis: number of visible patches per sample
    """
    B, N, C = x.shape
    len_keep = int(N * (1 - mask_ratio))
    if len_keep <= 0:
        len_keep = 1
    if len_keep >= N:
        len_keep = N - 1
    noise = torch.rand(B, N, device=x.device, dtype=x.dtype, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))
    mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
    mask.scatter_(1, ids_keep, True)
    return x_masked, mask, ids_restore, len_keep


def apply_mask_to_patches(
    patches: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    patches: (B, N, C), mask: (B, N) True = visible
    Returns patches with masked positions zeroed (for loss on masked only elsewhere).
    """
    mask_expand = mask.unsqueeze(-1).to(patches.dtype)
    return patches * (1 - mask_expand)
