import numpy as np
from src.config import MODEL_CONFIG


def tile_image(image, tile_size, stride):
    tiles = []
    positions = []

    H, W, C = image.shape

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):

            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions


def get_params(model_type):
    return MODEL_CONFIG[model_type]