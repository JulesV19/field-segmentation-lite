from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Data
    data_root: Path = Path("data/ftw")
    countries: list = field(default_factory=lambda: ["france", "netherlands", "spain", "belgium", "estonia", "latvia", "lithuania"])
    in_channels: int = 13  # 8 bandes S2 + 5 indices (NDVI×2, ΔNDVI, EVI×2)
    n_classes: int = 3     # 0=background, 1=interior, 2=border
    chip_size: int = 256   # resize en entrée (256 = natif)

    # Training
    batch_size: int = 12
    lr: float = 1e-4
    epochs: int = 30
    num_workers: int = 4

    # Loss : poids par classe [bg, interior, border]
    class_weights: list = field(default_factory=lambda: [0.5, 1.5, 3.0])

    # Checkpoints
    checkpoint_dir: Path = Path("checkpoints")
