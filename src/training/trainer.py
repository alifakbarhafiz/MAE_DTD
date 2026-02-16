"""Generic training utilities: optimizer step, LR scheduler, checkpoint saving."""
import torch
from pathlib import Path
from typing import Any, Optional


def get_optimizer(
    params,
    lr: float,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 1e-6,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    epoch: int = 0,
    extra: Optional[dict[str, Any]] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return {k: v for k, v in state.items() if k not in ("model", "optimizer")}
