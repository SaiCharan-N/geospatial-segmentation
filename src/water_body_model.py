import torch
import segmentation_models_pytorch as smp


def get_water_model():
    """
    Water segmentation model (DeepLabV3+ with ResNet34)

    ✔ Matches training configuration
    ✔ Supports GPU + FP16 inference
    ✔ Stable loading with saved weights
    """

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",   # ✅ MUST match training
        in_channels=3,
        classes=1,
        activation=None
    )

    return model