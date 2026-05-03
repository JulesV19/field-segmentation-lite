import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import FTWDataset
from models.segmodel import build_model
from postprocess.instance import predict_instances


# ── Métriques ─────────────────────────────────────────────────────────────────

def iou_per_class(pred: np.ndarray, target: np.ndarray, n_classes: int) -> list[float]:
    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        ious.append(float(inter) / float(union) if union > 0 else float("nan"))
    return ious


def instance_f1(pred_inst: np.ndarray, gt_inst: np.ndarray, iou_thr: float = 0.5) -> float:
    pred_ids = np.unique(pred_inst[pred_inst > 0])
    gt_ids   = np.unique(gt_inst[gt_inst > 0])
    tp = 0
    matched_gt = set()
    for pid in pred_ids:
        pred_mask = pred_inst == pid
        for gid in gt_ids:
            if gid in matched_gt:
                continue
            gt_mask = gt_inst == gid
            inter = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            if union > 0 and inter / union >= iou_thr:
                tp += 1
                matched_gt.add(gid)
                break
    fp   = len(pred_ids) - tp
    fn   = len(gt_ids)   - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ── Évaluation principale ──────────────────────────────────────────────────────

def evaluate(cfg: Config, checkpoint: str = "checkpoints/best.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = build_model(cfg.in_channels, cfg.n_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    test_ds = FTWDataset("france", "test", cfg.data_root, load_instances=True, chip_size=cfg.chip_size)
    loader  = DataLoader(test_ds, batch_size=1, num_workers=cfg.num_workers)

    all_ious = []
    all_f1s  = []
    # sample = (image, gt_mask, pred_mask, gt_inst, pred_inst, f1)
    samples  = []

    with torch.no_grad():
        for images, masks, instances in tqdm(loader, desc="Evaluation"):
            logits    = model(images.to(device))
            logits_np = logits.cpu().squeeze(0).numpy()   # (3, H, W)

            pred_mask = logits_np.argmax(axis=0)
            gt_mask   = masks.squeeze(0).numpy()
            all_ious.append(iou_per_class(pred_mask, gt_mask, cfg.n_classes))

            gt_inst   = instances.squeeze(0).numpy()
            pred_inst = predict_instances(logits_np)
            f1 = instance_f1(pred_inst, gt_inst)
            all_f1s.append(f1)

            if len(samples) < 8:
                samples.append((
                    images.squeeze(0).numpy(),
                    gt_mask, pred_mask,
                    gt_inst, pred_inst,
                    f1,
                ))

    mean_ious = np.nanmean(all_ious, axis=0)
    mean_f1   = float(np.mean(all_f1s))

    print(f"\n── Résultats sur le jeu de test ({len(test_ds)} chips) ──")
    print(f"  mIoU           : {np.nanmean(mean_ious):.4f}")
    print(f"    background   : {mean_ious[0]:.4f}")
    print(f"    intérieur    : {mean_ious[1]:.4f}")
    print(f"    bordure      : {mean_ious[2]:.4f}")
    print(f"  Instance F1@0.5: {mean_f1:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    _save_5col_grid(samples[:4], out_dir)
    _save_full_overlay(samples[:4], out_dir, mean_f1)
    _save_readme_overview(samples[:4], out_dir, mean_f1)

    print(f"  eval_samples.png      : grille 5 colonnes")
    print(f"  readme_overlay.png    : prédictions superposées sur RGB")
    print(f"  readme_overview.png   : vue 3 colonnes pour README")


# ── Helpers visuels ────────────────────────────────────────────────────────────

_CLASS_CMAP = mcolors.ListedColormap(["#111111", "#4daf4a", "#e41a1c"])

def _to_rgb(image: np.ndarray) -> np.ndarray:
    """B04=R, B03=G, B02=B, normalisés percentile 2-98."""
    rgb = np.stack([image[0], image[1], image[2]], axis=-1).astype(float)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

def _instance_rgb_plain(inst: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    out = np.zeros((*inst.shape, 3), dtype=float)
    for uid in np.unique(inst):
        if uid == 0:
            continue
        out[inst == uid] = rng.uniform(0.3, 1.0, size=3)
    return out

def _instance_overlay(rgb: np.ndarray, inst: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    rng = np.random.default_rng(42)
    out = rgb.copy()
    for uid in np.unique(inst):
        if uid == 0:
            continue
        color = rng.uniform(0.2, 1.0, size=3)
        mask  = inst == uid
        out[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return np.clip(out, 0, 1)

def _boundaries(inst: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Masque booléen des pixels de bordure entre instances."""
    edges = np.zeros(inst.shape, dtype=bool)
    for shift in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        shifted = np.roll(np.roll(inst, shift[0], axis=0), shift[1], axis=1)
        edges |= (inst != shifted) & (inst > 0)
    if thickness > 1:
        edges = binary_dilation(edges, iterations=thickness - 1)
    return edges

def _full_overlay(rgb: np.ndarray, inst: np.ndarray) -> np.ndarray:
    """Superpose les champs colorés + contours blancs sur l'image RGB."""
    out = _instance_overlay(rgb, inst, alpha=0.35)
    out[_boundaries(inst, thickness=2)] = (1.0, 1.0, 1.0)
    return np.clip(out, 0, 1)


# ── Grille 5 colonnes (debug) ──────────────────────────────────────────────────

def _save_5col_grid(samples: list, out_dir: Path):
    """
    Grille (5 cols × N lignes) : RGB | GT sem | Pred sem | GT inst | Pred inst
    Layout horizontal : chaque colonne = une vue, chaque ligne = un exemple.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4.2 * n),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.08})
    if n == 1:
        axes = axes[np.newaxis, :]

    for ax, t in zip(axes[0], ["RGB", "GT semantic", "Pred semantic", "GT instances", "Pred instances"]):
        ax.set_title(t, fontsize=10, fontweight="bold")

    for row, (image, gt_mask, pred_mask, gt_inst, pred_inst, f1) in enumerate(samples):
        rgb = _to_rgb(image)
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(gt_mask,   cmap=_CLASS_CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[row, 2].imshow(pred_mask, cmap=_CLASS_CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[row, 3].imshow(_instance_rgb_plain(gt_inst))
        axes[row, 4].imshow(_instance_rgb_plain(pred_inst))
        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].set_ylabel(f"F1={f1:.2f}", fontsize=9, rotation=0,
                                labelpad=32, va="center")

    fig.savefig(out_dir / "eval_samples.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


# ── Overlay complet (README) ───────────────────────────────────────────────────

def _save_full_overlay(samples: list, out_dir: Path, mean_f1: float):
    """
    Grille 2 lignes × N colonnes : ligne du haut = RGB, ligne du bas = overlay.
    Format paysage, compact pour le README.
    """
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 9),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})

    row_labels = ["Sentinel-2 input", f"Detected fields  (avg F1={mean_f1:.3f})"]

    for col, (image, gt_mask, pred_mask, gt_inst, pred_inst, f1) in enumerate(samples):
        rgb = _to_rgb(image)
        axes[0, col].imshow(rgb)
        axes[1, col].imshow(_full_overlay(rgb, pred_inst))
        for row in range(2):
            axes[row, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.02, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight="bold", color="white",
                          ha="right", va="center", rotation=90)

    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#0d1117")

    fig.savefig(out_dir / "readme_overlay.png", bbox_inches="tight",
                dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Vue 3 colonnes (README alternatif) ────────────────────────────────────────

def _save_readme_overview(samples: list, out_dir: Path, mean_f1: float):
    """
    Grille 3 lignes × N colonnes : RGB / ground truth / prédiction.
    Format paysage, compact pour le README.
    """
    n = len(samples)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 13),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})

    row_labels = ["Sentinel-2 (RGB)", "Ground truth", "Prediction"]

    for col, (image, gt_mask, pred_mask, gt_inst, pred_inst, f1) in enumerate(samples):
        rgb = _to_rgb(image)
        axes[0, col].imshow(rgb)
        axes[1, col].imshow(_full_overlay(rgb, gt_inst))
        axes[2, col].imshow(_full_overlay(rgb, pred_inst))
        for row in range(3):
            axes[row, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.02, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight="bold", color="white",
                          ha="right", va="center", rotation=90)
        axes[2, col].text(0.97, 0.03, f"F1={f1:.2f}",
                          transform=axes[2, col].transAxes,
                          fontsize=8, color="white", ha="right", va="bottom",
                          bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#0d1117")

    fig.savefig(out_dir / "readme_overview.png", bbox_inches="tight",
                dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    evaluate(Config())
