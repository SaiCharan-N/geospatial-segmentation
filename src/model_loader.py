import torch
import tensorflow as tf

from src.built_up_area_model import get_building_model
from src.road_model import get_road_model
from src.water_body_model import get_water_model
from src.water_line_model import get_water_line_model

from src.config import MODEL_CONFIG

# ✅ Enable TensorFlow GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


# 🔑 MODEL REGISTRY
MODEL_FACTORY = {
    "building": get_building_model,
    "road": get_road_model,
    "water": get_water_model,
    "water_line": get_water_line_model,
}


def load_model(model_type):

    config = MODEL_CONFIG[model_type]

    # =========================
    # 🔥 PYTORCH MODELS
    # =========================
    if config["framework"] == "torch":

        if model_type not in MODEL_FACTORY:
            raise ValueError(f"{model_type} not registered")

        model = MODEL_FACTORY[model_type]()

        model.load_state_dict(
            torch.load(config["model_path"], map_location="cpu"),
            strict=False
        )

        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

            if config.get("use_fp16", False):
                model = model.half()

        return model

    # =========================
    # 🔥 KERAS MODELS
    # =========================
    elif config["framework"] == "keras":

        model = tf.keras.models.load_model(
    config["model_path"],
    compile=False
)

        return model

    else:
        raise ValueError("Unknown framework")