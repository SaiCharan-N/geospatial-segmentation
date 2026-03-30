MODEL_CONFIG = {
    "building": {
        "tile_size": 512,
        "stride": 512,
        "min_pixels": 50,
        "model_path": "models/built_up_area.pth"
    },
    "road": {
        "tile_size": 512,
        "stride": 256,
        "min_pixels": 10,
        "model_path": "models/road.pth"
    },
    "water": {
        "tile_size": 512,
        "stride": 256,
        "min_pixels": None,
        "model_path": "models/water_body.pth"
    }
}