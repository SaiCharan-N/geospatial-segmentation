import torch

from src.built_up_area_model import get_building_model
from src.road_model import get_road_model
from src.water_body_model import get_water_model
from src.config import MODEL_CONFIG


def load_model(model_type):

    config = MODEL_CONFIG[model_type]

    if model_type == "building":
        model = get_building_model()

    elif model_type == "road":
        model = get_road_model()

    elif model_type == "water":
        model = get_water_model()

    model.load_state_dict(
        torch.load(config["model_path"], map_location="cpu"),
        strict=False   # IMPORTANT for ViT mismatch safety
    )

    model.eval()

# 🚀 FP16 for road (GPU only)
    if model_type == "road" and torch.cuda.is_available():
        model = model.half()

    return model