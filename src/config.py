MODEL_CONFIG = {

    "building": {
        "model_path": "models/built_up_area.pth",
        "framework": "torch",
        "use_fp16": False,
        "tile_size": 512,
        "stride": 256,
        "min_pixels": 300
    },

    "road": {
        "model_path": "models/road.pth",
        "framework": "torch",
        "use_fp16": True,
        "tile_size": 512,
        "stride": 256
    },

    "water": {
        "model_path": "models/water_body.pth",
        "framework": "torch",
        "use_fp16": False,
        "tile_size": 512,
        "stride": 256
    },

    # ✅ NEW
    "water_line": {
        "model_path": "models/water_resnet_unet.pth",
        "framework": "torch",
        "use_fp16": False,
        "tile_size": 256,     # 🔥 important (your training size)
        "stride": 128
    },

    # ✅ NEW
    "road_center": {
        "model_path": "models/road_center_line.keras",
        "framework": "keras",
        "tile_size": 256,     # 🔥 matches your keras model
        "stride": 128
    }
}