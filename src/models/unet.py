import torch
import torch.nn as nn


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloc Conv-BN-ReLU × 2, brique de base du U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    U-Net classique, 4 niveaux d'encodeur/décodeur.

    Flux pour une entrée (B, in_channels, 256, 256) :

    Encodeur                         Décodeur
    ────────────────────────────     ──────────────────────────────
    enc1 → (B,  64, 256, 256) ─────────────────────────→ dec1
    pool → (B,  64, 128, 128)                             ↑ concat + up1
    enc2 → (B, 128, 128, 128) ──────────────────→ dec2   |
    pool → (B, 128,  64,  64)                    ↑ concat + up2
    enc3 → (B, 256,  64,  64) ──────────→ dec3   |
    pool → (B, 256,  32,  32)            ↑ concat + up3
    enc4 → (B, 512,  32,  32) ──→ dec4   |
    pool → (B, 512,  16,  16)   ↑ concat + up4
    bottleneck (B, 1024, 16, 16)
    """

    def __init__(self, in_channels: int = 8, n_classes: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels  # 64

        # Encodeur
        self.enc1 = _double_conv(in_channels, c)
        self.enc2 = _double_conv(c,     c * 2)
        self.enc3 = _double_conv(c * 2, c * 4)
        self.enc4 = _double_conv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # Goulot d'étranglement
        self.bottleneck = _double_conv(c * 8, c * 16)
        self.drop = nn.Dropout2d(p=0.2)

        # Décodeur : Upsample bilinéaire + Conv (plus rapide que ConvTranspose2d sur MPS)
        self.up4   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(c * 16, c * 8, 1))
        self.dec4  = _double_conv(c * 16, c * 8)

        self.up3   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(c * 8, c * 4, 1))
        self.dec3  = _double_conv(c * 8, c * 4)

        self.up2   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(c * 4, c * 2, 1))
        self.dec2  = _double_conv(c * 4, c * 2)

        self.up1   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(c * 2, c, 1))
        self.dec1  = _double_conv(c * 2, c)

        # Tête de classification : 1×1 Conv → n_classes logits
        self.head = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodeur + skip connections
        s1 = self.enc1(x)                   # (B,  64, 256, 256)
        s2 = self.enc2(self.pool(s1))       # (B, 128, 128, 128)
        s3 = self.enc3(self.pool(s2))       # (B, 256,  64,  64)
        s4 = self.enc4(self.pool(s3))       # (B, 512,  32,  32)

        x  = self.drop(self.bottleneck(self.pool(s4))) # (B,1024,  16,  16)

        # Décodeur avec concatenation des skips
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))  # (B, 512, 32, 32)
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))  # (B, 256, 64, 64)
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))  # (B, 128,128,128)
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))  # (B,  64,256,256)

        return self.head(x)                  # (B, n_classes, 256, 256)
