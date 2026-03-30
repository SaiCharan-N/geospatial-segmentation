import segmentation_models_pytorch as smp
import torch.nn as nn


def get_building_model():

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,   # IMPORTANT (since you trained yourself)
        in_channels=3,
        classes=1,
        activation=None
    )

    return model