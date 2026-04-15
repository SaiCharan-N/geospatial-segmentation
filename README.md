Geospatial Feature Extraction using Multi-Model Deep Learning Pipeline

🚀 A scalable end-to-end geospatial AI system for extracting buildings, road networks, road centerlines, and water bodies from ultra-high-resolution satellite imagery and converting outputs into GIS-ready vector formats (.gpkg).

🏆 Hackathon Objective (IIT Tirupati)

To automate extraction of geospatial features from satellite imagery for:

🏙️ Smart city planning
🛣️ Road network analysis & mapping
🌊 Water resource monitoring
🌍 GIS-based spatial intelligence systems
🧠 System Overview (4-Model Geospatial AI Pipeline)

This project implements a modular multi-stage inference system designed for high-resolution satellite imagery.

🛰️ GeoTIFF Input (10k × 20k+ resolution)
        ↓
🔲 Tiling Engine (memory-efficient patch inference)
        ↓
🧠 Model Router (task-based selection)
        ↓
┌────────────────────────────────────────────┐
│            Deep Learning Models            │
│                                            │
│ 🏢 Model 1: Building Segmentation (UNet/CNN)  
│ 🛣️ Model 2: Road Segmentation (DINOv2-based)  
│ 🌊 Model 3: Water Segmentation (DeepLabV3+)  
│ 🌊 Model 4: Water Line Extraction Model (82 IoU)  
│ 🛣️ Model 5: Road Centerline Extraction (64 IoU)  
└────────────────────────────────────────────┘
        ↓
🧹 Post-processing Layer
- Morphological filtering
- Skeletonization (for centerlines)
- Contour extraction
- Polygonization

        ↓
🗺️ GIS Output Layer
- GeoTIFF masks
- GeoPackage (.gpkg)
🚀 Key Features
🧠 4-model specialized geospatial AI system
🛰️ Handles ultra-large satellite images (10k × 20k+)
🔲 Tile-based inference (memory-safe processing)
🔁 Multi-model routing architecture
🧭 Road centerline + road mask dual representation
🌊 Water segmentation + water boundary refinement
🗺️ GIS-ready vector export (.gpkg for QGIS/ArcGIS)
⚡ Optimized stitching + post-processing pipeline
🧠 Models & Performance
Feature	Model Type	Task	IoU
🏢 Buildings	CNN / UNet	Building footprint segmentation	~81%
🛣️ Roads	DINOv2-based model	Road surface segmentation	~79%
🌊 Water Bodies	DeepLabV3+	Water region segmentation	~82%
🌊 Water Line Model	Refinement CNN	Water boundary extraction	82%
🛣️ Road Centerline Model	Skeleton-based CNN	Road graph centerline extraction	64%
📂 Project Structure
Project/
│
├── data/
│   ├── input_images/
│   ├── output/
│
├── src/
│   ├── model_router.py
│   ├── inference_pipeline.py
│   ├── building_model.py
│   ├── road_model.py
│   ├── water_model.py
│   ├── water_line_model.py
│   ├── road_centerline_model.py
│   ├── utils.py
│
├── main.py
├── requirements.txt
└── README.md
⚙️ Installation
git clone https://github.com/SaiCharan-N/geospatial-segmentation.git
cd geospatial-segmentation

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
▶️ How to Run
python main.py

Workflow:

Select model or pipeline mode
Load GeoTIFF input
Run tiled inference
Generate segmentation outputs
Export GIS-ready files
📥 Input

Place satellite GeoTIFF files here:

data/input_images/

Example:

data/input_images/sample.tif
📤 Output

Generated outputs:

building_output.tif
road_output.tif
water_output.tif
water_line_output.tif
road_centerline_output.tif
.gpkg GIS vector files
⚡ Technical Innovations
🔲 Sliding window tiled inference for massive images
🧠 Multi-model routing architecture
🧭 Dual road representation (surface + centerline graph)
🌊 Water boundary refinement model
🧹 Skeletonization-based centerline extraction
🗺️ Raster → Vector GIS conversion pipeline
⚡ Memory-efficient inference for large-scale deployment
📌 Challenges Solved
Processing ultra-high-resolution satellite imagery (10k × 20k+)
Multi-model consistency across spatial outputs
Road centerline extraction from segmentation masks
Memory-safe inference under GPU constraints
Converting deep learning outputs into GIS-compatible formats
🚀 Future Improvements
🌐 Web-based GIS visualization dashboard
🧭 Road graph network extraction (shortest path analysis)
☁️ Cloud deployment (FastAPI + Docker)
📡 Real-time satellite inference API
⚡ Batch inference optimization pipeline
👨‍💻 Author

Sai Charan
AI & ML Engineer (Student)
Focus: Computer Vision | Geospatial AI | Deep Learning Systems

🏁 Final Impact Statement

This project demonstrates a production-grade geospatial intelligence system capable of converting raw satellite imagery into structured GIS data using a multi-model deep learning architecture, enabling scalable real-world deployment in urban planning and environmental monitoring.
