import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import LovaszLoss


class CombinedLoss(nn.Module):
    """
    0.5 × CrossEntropy(pondéré) + 0.5 × LovászLoss

    Lovász optimise directement le IoU (proxy différentiable),
    ce qui lui permet de casser les plateaux où Dice stagne.
    """

    def __init__(self, class_weights: list | None = None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

        self.lovasz = LovaszLoss(mode="multiclass", per_image=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce     = F.cross_entropy(logits, targets, weight=self.weight)
        lovasz = self.lovasz(logits, targets)
        return 0.5 * ce + 0.5 * lovasz
