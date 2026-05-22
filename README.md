# 🌊 Satellite Water Mapper

AI-based satellite image segmentation app that detects water regions using a deep learning U-Net model.

---

## 📌 Problem Statement

Satellite images contain large geographical areas where water bodies (rivers, lakes, oceans) are not clearly separated from land.

Manual detection is:
- Time-consuming
- Inconsistent
- Not scalable

We need an AI system that can automatically detect water regions from satellite images.

---

## 💡 Solution

This project uses a **U-Net deep learning model with ResNet34 encoder** to perform image segmentation.

It:
- Takes satellite images as input
- Processes them using a trained PyTorch model
- Generates a binary water mask
- Overlays prediction on original image

---

## 🧠 Model Architecture

- U-Net (Encoder-Decoder)
- Encoder: ResNet34
- Loss: Binary segmentation loss (trained separately)
- Framework: PyTorch + segmentation_models_pytorch

---

## 🛠️ Tools & Technologies

- Python
- PyTorch
- Torchvision
- Segmentation Models PyTorch (SMP)
- Streamlit
- NumPy
- PIL

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
