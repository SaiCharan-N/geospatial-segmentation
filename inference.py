import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import geopandas as gpd
from shapely.geometry import shape
import rasterio.features
import cv2
import tensorflow as tf   # ✅ NEW

from src.tiling_inference import get_params
from src.utils import preprocess_building, preprocess_road, preprocess_water
from src.preprocessing import remove_small_objects
from src.config import MODEL_CONFIG   # ✅ NEW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# PATCH SIZE REQUIREMENT
# =============================
def get_required_multiple(model_type):
    if model_type == "road":
        return 14
    elif model_type == "water":
        return 16
    elif model_type == "building":
        return 32
    else:
        return 16


# =============================
# PAD TO VALID SIZE
# =============================
def pad_to_multiple(img, multiple):
    h, w, c = img.shape

    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple

    pad_h = new_h - h
    pad_w = new_w - w

    padded = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='reflect'
    )

    return padded, h, w


# =============================
# 🔥 UPDATED PROCESS TILE (MULTI-FRAMEWORK)
# =============================
def process_tile(tile, model, model_type):

    config = MODEL_CONFIG[model_type]

    # -------- PREPROCESS --------
    if model_type == 'building':
        inp = preprocess_building(tile)
    elif model_type == 'road':
        inp = preprocess_road(tile)
    elif model_type == 'water':
        inp = preprocess_water(tile)
    else:
        inp = tile.astype(np.float32) / 255.0

    # -------- PAD --------
    multiple = get_required_multiple(model_type)
    inp, orig_h, orig_w = pad_to_multiple(inp, multiple)

    # =========================
    # 🔥 PYTORCH MODELS
    # =========================
    if config["framework"] == "torch":

        inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()

        if torch.cuda.is_available():
            inp_tensor = inp_tensor.to(DEVICE)

            if config.get("use_fp16", False):
                inp_tensor = inp_tensor.half()

        with torch.inference_mode():
            out = model(inp_tensor)
            out = torch.sigmoid(out)
            out = out.squeeze().cpu().numpy()

    # =========================
    # 🔥 KERAS MODELS
    # =========================
    elif config["framework"] == "keras":

        inp_keras = np.expand_dims(inp, axis=0)  # (1, H, W, C)
        out = model.predict(inp_keras, verbose=0)[0, :, :, 0]

    else:
        raise ValueError(f"Unknown framework for {model_type}")

    # -------- CROP BACK --------
    out = out[:orig_h, :orig_w]

    return out


# =============================
# MAIN PIPELINE
# =============================
def predict_large_image(image_path, model, model_type, output_tif, output_gpkg):

    config = MODEL_CONFIG[model_type]

    # ✅ HANDLE DEVICE ONLY FOR PYTORCH
    if config["framework"] == "torch":
        model = model.to(DEVICE)
        model.eval()

        if config.get("use_fp16", False) and torch.cuda.is_available():
            model = model.half()

    config_params = get_params(model_type)
    tile_size = config_params['tile_size']
    stride = config_params['stride']
    min_pixels = config_params.get('min_pixels')

    # -------- SPEED OPTIMIZATION --------
    if model_type == "road":
        stride = int(tile_size * 0.75)
    if model_type == "water":
        stride = int(tile_size * 0.75)

    with rasterio.open(image_path) as src:

        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform

        counts = np.zeros((H, W), dtype=np.uint16)
        accum = np.zeros((H, W), dtype=np.float32)

        print(f"Processing {H}x{W} with tile {tile_size}, stride {stride}...")

        for y in range(0, H, stride):
            for x in range(0, W, stride):

                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)

                window = Window(x, y, x_end - x, y_end - y)
                tile = src.read(window=window)
                tile = np.transpose(tile, (1, 2, 0))

                # -------- CHANNEL FIX --------
                if tile.shape[-1] > 3:
                    tile = tile[..., :3]
                elif tile.shape[-1] == 1:
                    tile = np.repeat(tile, 3, axis=-1)

                # -------- SKIP EMPTY (ROAD ONLY) --------
                if model_type == "road":
                    if np.mean(tile) < 15:
                        continue

                # -------- EDGE PADDING --------
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]

                    tile = np.pad(
                        tile,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='reflect'
                    )

                    pred = process_tile(tile, model, model_type)
                    pred = pred[:y_end - y, :x_end - x]
                else:
                    pred = process_tile(tile, model, model_type)

                accum[y:y_end, x:x_end] += pred
                counts[y:y_end, x:x_end] += 1

        # -------- AVERAGING --------
        print("Averaging predictions...")
        mask = accum / np.maximum(counts, 1)

        # -------- THRESHOLD + POST --------
        if model_type == "road":
            mask = (mask > 0.25).astype(np.uint8)
            mask = remove_small_objects(mask, 150)

        elif model_type == "water":
            mask = (mask > 0.4).astype(np.uint8)

            mask = cv2.medianBlur(mask, 5)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            mask = remove_small_objects(mask, 500)

        else:
            mask = (mask > 0.3).astype(np.uint8)
            if min_pixels:
                mask = remove_small_objects(mask, min_pixels)

        # -------- SAVE TIF --------
        print("Saving GeoTIFF...")
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(mask * 255, 1)

        # -------- VECTORIZE --------
        print("Generating GPKG...")
        shapes_gen = rasterio.features.shapes(mask.astype(np.uint8), transform=transform)

        polygons = [shape(geom) for geom, val in shapes_gen if val == 1]

        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
            gdf["geometry"] = gdf["geometry"].simplify(0.5)
            gdf.to_file(output_gpkg, driver="GPKG")
            print(f"Saved {len(polygons)} polygons → {output_gpkg}")
        else:
            print("No objects detected.")

    return True