# 3D Spatial Scene Reconstruction Engine

A high-performance local pipeline utilizing 3D Gaussian Splatting and CUDA acceleration to reconstruct photorealistic 3D environments from sequential spatial frames.

---



---

### 📌 What Problem It Solves
Traditional 3D reconstruction pipelines (like standard Structure-from-Motion combined with Multi-View Stereo) are computationally heavy, take hours to process, and struggle to capture fine, complex geometries like thin structures, reflections, specularities, and semi-transparent surfaces. 

This engine solves that by replacing dense voxel or mesh generation with **3D Gaussian Splatting**. It optimizes a sparse tracking matrix into millions of continuous, differentiable 3D Gaussians. By skipping heavy geometry mesh building and using direct rasterization instead, it achieves **real-time, photorealistic novel view synthesis** with extremely fast training convergence.

---

### 📊 Results
* **Dataset Size:** 191 high-resolution spatial frames
* **Peak Signal-to-Noise Ratio (PSNR):** ~28.4 dB *(Optimal convergence baseline)*
* **Structural Similarity Index (SSIM):** ~0.912
* **Real-Time Render Speed:** 100+ FPS at 1080p resolution post-optimization

---

### 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

---

### 🚀 How to Run Locally

#### 1. Clone the Repository
bash
git clone [https://github.com/yourusername/3D-Scene-Reconstruction-Engine.git](https://github.com/yourusername/3D-Scene-Reconstruction-Engine.git)
cd 3D-Scene-Reconstruction-Engine
####  2. Install Dependencies & Custom CUDA Kernels
terminal/bash
 pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
####  3. Run SfM Feature Tracking (COLMAP)
terminal/bash
colmap feature_extractor --database_path custom_data/database.db --image_path custom_data/images --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path custom_data/database.db
colmap mapper --database_path custom_data/database.db --image_path custom_data/images --output_path custom_data/sparse/0
####  4. Train the Gaussian Splatting Model
terminal/bash
 python train.py -s custom_data --model_path output/reconstruction_result
git clone [https://github.com/yourusername/3D-Scene-Reconstruction-Engine.git](https://github.com/yourusername/3D-Scene-Reconstruction-Engine.git)
cd 3D-Scene-Reconstruction-Engine
