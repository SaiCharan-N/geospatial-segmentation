# 🛰️ Geospatial Feature Extraction using Deep Learning

🚀 End-to-end pipeline for extracting buildings, roads, and water bodies from large satellite images and converting them into GIS-ready vector data.

---

## 📌 Problem Statement

Manual extraction of geospatial features such as buildings, roads, and water bodies from satellite imagery is time-consuming and not scalable.

This project automates feature extraction using deep learning to support:

* 🏙️ Urban planning
* 🛣️ Infrastructure development
* 🌊 Environmental monitoring

The system processes large GeoTIFF images and generates both raster and vector outputs suitable for GIS applications.

---

## 🚀 Try It Yourself (Real Data + Outputs)

👉 **Download full-resolution input & predictions:**
https://drive.google.com/drive/folders/1VUOgMMm7exBNOA2OdNyFAkyRkjgDfj8L?usp=sharing

Includes:

* 🛰️ High-resolution satellite GeoTIFF (~300MB)
* 🏢 Building segmentation results
* 🛣️ Road extraction outputs
* 🌊 Water body predictions
* 📍 Vector shapefiles (.gpkg)
(THESE MASKS AND SHAPE FILES ARE GENERATED USING OUR MODELS)

💡 You can directly open the `.gpkg` files in QGIS to visualize extracted features.

⚠️ Hosted externally due to GitHub file size limits.

---

## 📊 Results

* ✅ Successfully processed **10k × 20k+ resolution satellite images**
* ⚡ Efficient tiling-based inference without memory overflow
* 🎯 High-quality segmentation performance:

  * 🏢 Building IoU: **81%**
  * 🛣️ Road IoU: **79%**
  * 🌊 Water IoU: **79%**
* 🗺️ Generated GIS-compatible vector outputs (GeoPackage)

These outputs can be directly used in mapping and planning tools.

---

## 🚀 Key Features

* 🏢 Accurate Building Segmentation
* 🛣️ Road Network Extraction using DINOv2
* 🌊 Water Body Detection using DeepLabV3+
* 🧠 Handles ultra high-resolution satellite imagery
* 🗺️ Outputs both raster masks and vector shapefiles

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

python -m venv venv
venv\Scripts\activate   # Windows

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

Place your GeoTIFF file inside:

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
* Corresponding `.gpkg` vector files

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

## 🧰 Tech Stack

* Python
* PyTorch
* segmentation-models-pytorch
* Rasterio
* GeoPandas
* OpenCV
* NumPy

---

## ⚡ Optimizations

* Tiling-based inference for large images
* FP16 inference for faster road detection
* Dynamic padding for model compatibility
* Overlap-based stitching
* Noise removal & polygon refinement

---

## 📊 Sample Outputs

Sample outputs are available in:

```
data/output/
```

For full-resolution results:
👉 Refer to the Google Drive link above

---

## 🧩 Challenges Solved

* Handling **huge satellite images (10k × 20k+)**
* Managing different model constraints (patch sizes)
* Efficient tiling + stitching
* Memory-safe inference
* Clean vector (GIS-ready) generation

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
