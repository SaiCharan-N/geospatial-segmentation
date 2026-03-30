# 🛰️ Geospatial Feature Extraction using Deep Learning

This project performs **automatic extraction of geospatial features** (Buildings, Roads, Water Bodies) from large satellite images using deep learning models.

---
## 🚀 Try It Yourself (Input + Outputs)

📦 **Download Sample Data & Results (GeoTIFF + GPKG):**
👉 https://drive.google.com/drive/folders/1VUOgMMm7exBNOA2OdNyFAkyRkjgDfj8L?usp=sharing

This folder contains:

* 🛰️ Sample input GeoTIFF image
* 🏢 Building segmentation output
* 🛣️ Road extraction output
* 🌊 Water body detection output
* 📍 Vector shapefiles (.gpkg)

⚠️ Note: Files are large and hosted externally due to GitHub size limits.

## 🚀 Features

* ✅ Building Segmentation
* 🛣️ Road Extraction (DINOv2-based model)
* 🌊 Water Body Detection (DeepLabV3+)
* 🧠 Handles **large GeoTIFF images (10k+ resolution)**
* 🗺️ Generates:

  * Raster mask (`.tif`)
  * Vector output (`.gpkg`)

---

## 📂 Project Structure

```
Project/
│
├── data/
│   ├── input_images/
│   │   └── sample.tif
│   └── output/
│       ├── building_output.tif
│       ├── road_output.tif
│       └── water_output.tif
│
├── src/
│   ├── model_loader.py
│   ├── inference.py
│   ├── road_model.py
│   ├── water_body_model.py
│   ├── building_model.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

You will see:

```
Select model:
1. building
2. road
3. water
```

Enter your choice.

---

## 📥 Input

Place your input GeoTIFF inside:

```
data/input_images/
```

Example:

```
data/input_images/sample.tif
```

---

## 📤 Output

Outputs are generated in the project root:

* `building_output.tif`
* `road_output.tif`
* `water_output.tif`
* Corresponding `.gpkg` files

---

## 🧠 Models Used

### 🏢 Building Model

* CNN / UNet-based segmentation

### 🛣️ Road Model

* DINOv2 (Vision Transformer)
* Lightweight custom decoder

### 🌊 Water Model

* DeepLabV3+
* ResNet34 backbone

---

## ⚡ Optimizations

* Tiling-based inference for large images
* FP16 inference for faster road detection
* Dynamic padding (handles model constraints)
* Overlap-based stitching
* Noise removal & polygon simplification

---

## 📊 Sample Outputs

Sample outputs are available in:

```
data/output/
```

Includes:

* Raster masks
* Vectorized shapefiles (GeoPackage)

---

## 🧩 Challenges Solved

* Handling **huge satellite images (10k x 20k+)**
* Managing different model constraints (patch sizes)
* Efficient tiling + stitching
* Memory-safe inference
* Clean vector generation

---

## 📌 Future Improvements

* Batch tile inference (3–5× faster)
* Road network graph extraction
* Web-based visualization
* Deployment using Flask / FastAPI

---

## 👨‍💻 Author

**Sai Charan**

* AI & ML Student
* Focus: Computer Vision, Geospatial AI

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub
