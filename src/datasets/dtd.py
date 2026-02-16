"""
DTD (Describable Textures Dataset) loader.
Handles splits, augmentations; returns images + labels.
Dataset logic isolated from training.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# DTD has 47 texture categories
DTD_NUM_CLASSES = 47


class DTDDataset(Dataset):
    """
    DTD dataset. Expects standard layout:
      root/
        images/
          banded/  ... (folder per class)
        labels/
          train1.txt, val1.txt, test1.txt  (each line: images/classname/imagename.jpg)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_index: int = 1,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.split_index = split_index
        self.image_size = image_size
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []  # (path, label)
        self.class_to_idx: dict[str, int] = {}
        self._load_split()

    def _load_split(self) -> None:
        labels_dir = self.root / "labels"
        images_dir = self.root / "images"
        if not labels_dir.exists():
            labels_dir = self.root
            images_dir = self.root
        split_file = labels_dir / f"{self.split}{self.split_index}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"DTD split file not found: {split_file}. "
                "Ensure data/dtd/labels/ has train1.txt, val1.txt, test1.txt"
            )
        # Build class_to_idx from folder names
        if (self.root / "images").exists():
            classes = sorted([d.name for d in (self.root / "images").iterdir() if d.is_dir()])
        else:
            classes = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        if len(self.class_to_idx) == 0 and (self.root / "labels").exists():
            # Infer from first line of split file: images/classname/file.jpg
            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        cls = parts[-2]
                        if cls not in self.class_to_idx:
                            self.class_to_idx[cls] = len(self.class_to_idx)
            self.class_to_idx = {c: i for i, c in enumerate(sorted(self.class_to_idx))}
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.replace("\\", "/")
                full_path = self.root / rel_path
                if not full_path.exists():
                    full_path = self.root / "images" / rel_path
                if not full_path.exists():
                    continue
                parts = rel_path.split("/")
                if len(parts) >= 2:
                    cls = parts[-2]
                else:
                    cls = "unknown"
                label = self.class_to_idx.get(cls, 0)
                self.samples.append((str(full_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_dtd_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    split_index: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    """
    train_ds = DTDDataset(
        root=root,
        split="train",
        split_index=split_index,
        image_size=image_size,
        transform=_train_transform(image_size),
    )
    val_ds = DTDDataset(
        root=root,
        split="val",
        split_index=split_index,
        image_size=image_size,
        transform=_eval_transform(image_size),
    )
    test_ds = DTDDataset(
        root=root,
        split="test",
        split_index=split_index,
        image_size=image_size,
        transform=_eval_transform(image_size),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
