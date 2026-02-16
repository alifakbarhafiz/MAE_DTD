"""Logging and experiment directory setup."""
import os
import sys
from pathlib import Path
from datetime import datetime


def get_log_dir(base_dir: str, run_name: str) -> Path:
    """Create and return a timestamped log directory for this run."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{run_name}_{timestamp}"
    log_dir = base / name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_ckpt_dir(base_dir: str, run_name: str, create: bool = True) -> Path:
    """Return checkpoint directory for this run (optionally create)."""
    base = Path(base_dir)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    ckpt_dir = base / run_name
    if create:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def tee_log(log_path: Path):
    """Context manager: stdout also written to log_path."""
    class Tee:
        def __init__(self, path):
            self.path = path
            self.file = None
            self.stdout = sys.stdout

        def __enter__(self):
            self.file = open(self.path, "w", encoding="utf-8")
            sys.stdout = self
            return self

        def __exit__(self, *args):
            sys.stdout = self.stdout
            if self.file:
                self.file.close()

        def write(self, data):
            self.stdout.write(data)
            if self.file:
                self.file.write(data)
            self.flush()

        def flush(self):
            self.stdout.flush()
            if self.file:
                self.file.flush()

    return Tee(log_path)
