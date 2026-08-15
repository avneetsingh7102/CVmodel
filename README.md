# Live Computer Vision Project (ResNet-50 CIFAR-100)

A computer vision application powered by a fine-tuned **ResNet-50** deep learning model trained on **CIFAR-100**. Supports real-time webcam inference, desktop OpenCV window with dynamic HUD, image snapshot saving, and an interactive FastAPI Web Computer Vision Studio.

---

## 🚀 Features

- **Trained ResNet-50 Model Integration**: Loads `resnet50_cifar100_finetuned.pth` with hardware acceleration auto-detection (**Apple Silicon Metal MPS**, **NVIDIA CUDA**, or **CPU**).
- **Desktop OpenCV Feed (`--mode desktop`)**:
  - Live webcam stream with real-time target region box.
  - Interactive Heads-Up Display (HUD) with Top-5 probability horizontal bar charts.
  - Keybindings: `c` (predict frame), `r` (toggle continuous real-time mode), `s` (save snapshot), `q` (quit).
- **Web Computer Vision Studio (`--mode web`)**:
  - Web UI powered by FastAPI and Uvicorn.
  - Real-time webcam classification directly in browser.
  - Drag-and-Drop Image Uploader for testing static files.
  - Glassmorphic dark theme UI with animated confidence indicators.
- **Single Image Predictor (`predict_image.py`)**:
  - CLI command to classify any image file.

---

## 📂 Project Files

- [main.py](file:///Users/avneetsingh/Documents/AI%20projects%20/Forge%20CLI/CV%20project/main.py): Unified CLI entry point for Desktop and Web modes.
- [desktop_cv.py](file:///Users/avneetsingh/Documents/AI%20projects%20/Forge%20CLI/CV%20project/desktop_cv.py): Desktop OpenCV webcam application with HUD overlays.
- [web_app.py](file:///Users/avneetsingh/Documents/AI%20projects%20/Forge%20CLI/CV%20project/web_app.py): FastAPI web server and single-page Web Vision Studio.
- [model_loader.py](file:///Users/avneetsingh/Documents/AI%20projects%20/Forge%20CLI/CV%20project/model_loader.py): Model loading, preprocessing pipeline, and inference engine.
- [cifar100_labels.py](file:///Users/avneetsingh/Documents/AI%20projects%20/Forge%20CLI/CV%20project/cifar100_labels.py): Official 100 CIFAR-100 class names and supercategory mapping.
- [predict_image.py](file:///Users/avneetsingh/Documents/AI%20projects%20/Forge%20CLI/CV%20project/predict_image.py): CLI tool for single image classification.
- `resnet50_cifar100_finetuned.pth`: Fine-tuned PyTorch ResNet-50 weights.

---

## 🛠️ Requirements & Setup

Make sure PyTorch, torchvision, OpenCV, and FastAPI are installed:

```bash
pip install torch torchvision opencv-python pillow fastapi uvicorn
```

---

## 🎮 How to Run

### 1. Run Desktop Webcam Mode (OpenCV Window)

```bash
python3 main.py --mode desktop
```

- Press `c`: Capture & predict current frame
- Press `r`: Toggle continuous real-time inference
- Press `s`: Save snapshot with HUD overlay to `snapshots/` folder
- Press `q`: Quit application

---

### 2. Run Web Studio Server (Browser Mode)

```bash
python3 main.py --mode web --port 8000
```

Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the interactive web interface.

---

### 3. Predict Single Image File

```bash
python3 predict_image.py /path/to/your/image.jpg
```
