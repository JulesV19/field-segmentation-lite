import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import Config
from data.dataset import FTWDataset
from data.transforms import get_train_transform
from models.segmodel import build_model
from models.losses import CombinedLoss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(cfg: Config):
    device = get_device()
    print(f"Device : {device}")

    train_ds = ConcatDataset([
        FTWDataset(c, "train", cfg.data_root, transform=get_train_transform(), chip_size=cfg.chip_size)
        for c in cfg.countries
    ])
    val_ds = ConcatDataset([
        FTWDataset(c, "val", cfg.data_root, chip_size=cfg.chip_size)
        for c in cfg.countries
    ])
    print(f"Pays : {cfg.countries}")
    print(f"Train : {len(train_ds)} chips  |  Val : {len(val_ds)} chips")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, pin_memory=False,
                              persistent_workers=True)

    model = build_model(cfg.in_channels, cfg.n_classes).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),           "lr": cfg.lr * 0.3},
        {"params": model.decoder.parameters(),           "lr": cfg.lr},
        {"params": model.segmentation_head.parameters(), "lr": cfg.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = CombinedLoss(cfg.class_weights).to(device)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # ── Entraînement ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.epochs} [train]", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
            loss = criterion(logits.float(), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        iou_inter = np.zeros(cfg.n_classes)
        iou_union = np.zeros(cfg.n_classes)
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{cfg.epochs} [val]  ", leave=False):
                images = images.to(device)
                masks  = masks.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(images)
                val_loss += criterion(logits.float(), masks).item()

                pred = logits.argmax(dim=1).cpu().numpy()
                gt   = masks.cpu().numpy()
                for c in range(cfg.n_classes):
                    iou_inter[c] += ((pred == c) & (gt == c)).sum()
                    iou_union[c] += ((pred == c) | (gt == c)).sum()

        if device.type == "mps":
            torch.mps.synchronize()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        ious = [i / u if u > 0 else np.nan for i, u in zip(iou_inter, iou_union)]
        miou = float(np.nanmean(ious))
        int_iou = float(ious[1]) if not np.isnan(ious[1]) else 0.0
        brd_iou = float(ious[2]) if not np.isnan(ious[2]) else 0.0
        # Interior sépare champ/non-champ ; border sépare champs adjacents.
        # Les deux sont nécessaires pour un bon Instance F1.
        score = 0.6 * int_iou + 0.4 * brd_iou
        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mIoU={miou:.4f}  [bg={ious[0]:.3f} int={ious[1]:.3f} brd={ious[2]:.3f}]")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")
            print(f"             ✓ checkpoint sauvegardé (int={int_iou:.4f} brd={brd_iou:.4f})")

    print("Entraînement terminé.")


if __name__ == "__main__":
    train(Config())
