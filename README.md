# CubeSat Synthetic Video Generation & Computer Vision Evaluation

This repository was developed by Giovanna Girotto for a Semester thesis focused on generating synthetic space imagery of CubeSats and evaluating computer vision algorithms for feature detection, matching, and tracking.

It consists of two main components:
1. **Synthetic Data Generation** – rendering videos of CubeSats in realistic space environments using Blender.
2. **Image Processing & Analysis** – applying CV algorithms (e.g., SIFT, ORB, KLT) to the generated videos to benchmark performance.

---

## Repository Structure

```
fpemtpy/
├── src/
│   ├── synthetic_video_generation/   # Blender video generation pipeline
│   └── image_processing/             # Computer vision evaluation scripts
├── templates/                        # 3D models & backgrounds for Blender
├── results/                          # Output videos and processed CV metrics
├── support/                          # Graph generation, Excel input, etc.
└── README.md                         # This file
```

---

## 🔧 What This Repository Does

### 1. Synthetic CubeSat Video Generation
- Uses Blender’s Python API to render animated videos of CubeSats flying in space.
- Allows variation in camera angle, lighting, CubeSat model, background, and noise.
- Designed for batch rendering to generate large datasets for controlled testing.

See: [`src/synthetic_video_generation/README_blender.md`](src/synthetic_video_generation/README_blender.md)

---

### 2. Computer Vision Processing
- Applies feature detection and tracking algorithms (SIFT, ORB, AKAZE, KLT, etc.) to the synthetic videos.
- Computes metrics like error magnitude, outlier ratios, keypoints in ROI, and more.
- Stores aggregated results in Excel files for analysis.

See: [`src/image_processing/README_CV.md`](src/image_processing/README_CV.md)

---

## 📁 Folder Overview

| Folder | Description |
|--------|-------------|
| `src/synthetic_video_generation/` | Scripts for generating synthetic CubeSat videos with Blender. |
| `src/image_processing/`          | Scripts for analyzing videos with CV algorithms and collecting metrics. |
| `templates/`                     | Assets used for rendering: CubeSat `.blend` models and background space images. |
| `results/`                       | Output folder for: <br> - Rendered CubeSat videos (`videos/`) <br> - Excel files with computed CV metrics. |
| `support/`                       | Additional files like plotting tools, Excel inputs, helper scripts for result visualization. |

---

## Results

All generated videos and corresponding analysis data are stored in the `results/` directory:
- `results/videos/` – Contains rendered `.mp4` videos of CubeSats.
- `results/methods_data.xlsx` (or similar) – Stores metrics computed for each video and algorithm.


