import segmentation_models_pytorch as smp


def build_model(in_channels: int, n_classes: int):
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=n_classes,
        activation=None,
    )
