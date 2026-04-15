from src.model_loader import load_model
from inference import predict_large_image
import os

IMAGE_PATH = r"C:\Users\HP\Desktop\Project\data\input_images\Badetumnar.tif"


if __name__ == "__main__":

    print("\nSelect model:")
    print("1. building")
    print("2. road")
    print("3. water")
    print("4. water_line")        # ✅ NEW
    print("5. road_center")       # ✅ NEW

    choice = input("Enter choice: ").strip()

    model_map = {
        "1": "building",
        "2": "road",
        "3": "water",
        "4": "water_line",        # ✅ NEW
        "5": "road_center"        # ✅ NEW
    }

    if choice not in model_map:
        print("Invalid choice, exiting.")
        exit(1)

    model_type = model_map[choice]

    print(f"\nLoading {model_type} model...")
    model = load_model(model_type)

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find image at {IMAGE_PATH}")
        exit(1)

    # ✅ Better output naming
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    output_tif = f"{base_name}_{model_type}.tif"
    output_gpkg = f"{base_name}_{model_type}.gpkg"

    print("\nRunning Inference & Post-processing pipeline...")
    print(f"Model: {model_type}")
    print(f"Input: {IMAGE_PATH}")
    print(f"Output TIF: {output_tif}")
    print(f"Output GPKG: {output_gpkg}")

    predict_large_image(
        IMAGE_PATH,
        model,
        model_type,
        output_tif,
        output_gpkg
    )

    print("\n✅ Done!")