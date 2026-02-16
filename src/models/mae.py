"""
Masked Autoencoder (MAE).
Handles masking, encoder (visible only), decoder, reconstruction loss.
"""

import torch
import torch.nn as nn
import numpy as np

from .vit_encoder import ViTEncoder
from ..utils.masking import random_masking


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """imgs: (B, C, H, W) -> (B, N, C*patch_size^2)."""
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h, w = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, C * patch_size * patch_size)
    return x


def unpatchify(x: torch.Tensor, patch_size: int, C: int = 3) -> torch.Tensor:
    """x: (B, N, C*patch_size^2) -> (B, C, H, W)."""
    B, N, _ = x.shape
    h = w = int(np.sqrt(N))
    x = x.reshape(B, h, w, patch_size, patch_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, h * patch_size, w * patch_size)
    return x


class MAE(nn.Module):
    """
    Masked Autoencoder: encoder sees only visible patches; decoder reconstructs
    all patches; loss on masked patches only.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm_pix_loss = norm_pix_loss
        self.num_patches = (img_size // patch_size) ** 2

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # Encoder patch pos_embed (no CLS): use encoder's pos_embed for patches only
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * 4),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns loss (scalar), pred (B, N, p*p*3), mask (B, N) where True = masked.
        """
        B, C, H, W = imgs.shape
        patches = self.encoder.patch_embed(imgs)
        x_masked, mask, ids_restore, len_keep = random_masking(patches, mask_ratio, generator=generator)
        pos_vis = self._gather_pos(self.encoder_pos_embed, ids_restore, len_keep)
        latent = self.encoder.forward_visible(x_masked, pos_vis)
        full_latent = self._full_seq_with_mask_tokens(latent, mask, ids_restore)
        pred = self.decode(full_latent)
        target = patchify(imgs, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True) + 1e-6
            target = (target - mean) / var.sqrt()
        loss = self._loss_on_masked(pred, target, mask)
        return loss, pred, mask

    def _gather_pos(
        self,
        pos_embed: torch.Tensor,
        ids_restore: torch.Tensor,
        len_keep: int,
    ) -> torch.Tensor:
        """Position embedding for visible tokens only. pos_embed (1, N, D)."""
        ids_keep = ids_restore[:, :len_keep]
        return torch.gather(pos_embed.expand(ids_keep.shape[0], -1, -1), 1, ids_keep.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1]))

    def _full_seq_with_mask_tokens(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct full sequence: [visible latents | mask tokens] then reorder to original patch order."""
        B, len_keep, _ = latent.shape
        N_full = mask.shape[1]
        dec_dim = self.mask_token.shape[-1]
        mask_tokens = self.mask_token.expand(B, N_full - len_keep, -1)
        full = torch.cat([self.decoder_embed(latent), mask_tokens], dim=1)
        ids_reorder = torch.argsort(ids_restore, dim=1)
        full = torch.gather(full, 1, ids_reorder.unsqueeze(-1).expand(-1, -1, dec_dim))
        return full

    def decode(self, full_latent: torch.Tensor) -> torch.Tensor:
        """full_latent (B, N, decoder_embed_dim) -> pred (B, N, p*p*C)."""
        x = full_latent + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def _loss_on_masked(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Loss only on masked patches. mask True = was masked."""
        pred_masked = pred[mask]
        target_masked = target[mask]
        return (pred_masked - target_masked).pow(2).mean()
