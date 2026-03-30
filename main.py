from src.model_loader import load_model
from inference import predict_large_image
import os
IMAGE_PATH = os.path.join("data", "input_images", "sample.tif")
if __name__ == "__main__":
    print("Select model:")
    print("1. building")
    print("2. road")
    print("3. water")

    choice = input("Enter choice: ")

    model_map = {
        "1": "building",
        "2": "road",
        "3": "water"
    }

    if choice not in model_map:
        print("Invalid choice, exiting.")
        exit(1)

    model_type = model_map[choice]

    print(f"Loading {model_type} model...")
    model = load_model(model_type)

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find image at {IMAGE_PATH}")
        exit(1)

    output_tif = f"{model_type}_output.tif"
    output_gpkg = f"{model_type}_output.gpkg"

    print("Running Inference & Post-processing pipeline (GeoTIFF + GPKG generation)...")
    predict_large_image(IMAGE_PATH, model, model_type, output_tif, output_gpkg)

    print("Done!")
