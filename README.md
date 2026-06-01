# 🔥 Fire Detection & People Counting System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20Desktop-lightgrey.svg)]()
![Tests](https://github.com/ALEN2002-py/Fire-Detection-People-Counting/actions/workflows/python-app.yml/badge.svg)

A real-time computer vision system for **fire detection** using a fine-tuned InceptionV3 deep learning model combined with **people counting** using HOG + SVM descriptors. Designed for deployment on edge devices such as Raspberry Pi for smart building safety applications.

---

## 📌 Overview

This system addresses a critical safety gap: delayed detection of fires and uncertain occupancy counts during emergencies. It provides two independently operable but complementary modules:

| Module | Approach | Input | Output |
|---|---|---|---|
| **Fire Detection** | Transfer Learning (InceptionV3) | Camera feed / image | Fire / No-Fire classification + confidence |
| **People Counting** | HOG + LinearSVM (OpenCV) | Camera feed / image / video | Bounding boxes + headcount |

The combined system can alert emergency services while simultaneously reporting the number of people present, enabling faster, more informed emergency response.

---

## 🗂️ Repository Structure

```
fire-detection-people-counting/
│
├── fire_detection/
│   ├── train.ipynb                  # Full training pipeline (Google Colab)
│   └── inference.py                 # Real-time fire detection from webcam
│
├── people_counting/
│   └── detector.py                  # HOG-based people detection (image/video/webcam)
│
├── docs/
│   ├── architecture.md              # System design and model architecture details
│   └── results.md                   # Training metrics and evaluation results
│
├── tests/
│   └── test_inference.py            # Unit tests for inference modules
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 🧠 Technical Architecture

### Module 1 — Fire Detection (InceptionV3 Transfer Learning)

The fire detection module uses **Transfer Learning** on the **InceptionV3** architecture, pre-trained on ImageNet. Only the classification head is initially trained; subsequently, the top layers are unfrozen for fine-tuning.

```
Input Image (224×224×3)
        │
        ▼
InceptionV3 Base (frozen, ImageNet weights)
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dense(2048, ReLU) → Dropout(0.25)
        │
        ▼
Dense(1024, ReLU) → Dropout(0.20)
        │
        ▼
Dense(2, Softmax)  ──►  [Fire | No-Fire]
```

**Training Strategy:**
1. **Phase 1 — Feature extraction:** Freeze all base layers, train only the custom head with RMSprop.
2. **Phase 2 — Fine-tuning:** Unfreeze layers from layer 249 onward, retrain with a low SGD learning rate (lr=0.0001, momentum=0.9) to avoid destroying pre-trained weights.

**Dataset:** [DeepQuestAI Fire-Smoke Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset) (Fire / No-Fire binary classification; Smoke class excluded to reduce ambiguity)

**Data Augmentation:** Zoom (±15%), horizontal flip, nearest-neighbour fill — applied to training set only.

---

### Module 2 — People Counting (HOG + LinearSVM)

The people counting module uses OpenCV's built-in **Histogram of Oriented Gradients (HOG)** descriptor with a pre-trained **Linear SVM** (`cv2.HOGDescriptor_getDefaultPeopleDetector()`).

```
Input Frame
    │
    ▼
Resize (max width 800px, aspect-ratio preserved)
    │
    ▼
HOG Feature Extraction (winStride=4×4, padding=8×8, scale=1.03)
    │
    ▼
LinearSVM Sliding Window Detection
    │
    ▼
Bounding Boxes → Annotated Frame + Count Overlay
```

**Why HOG+SVM over a neural detector?**
HOG+SVM is deterministic, runs without a GPU, and is well-suited for low-power edge devices (Raspberry Pi). For higher accuracy requirements, this module can be swapped for a MobileNet-SSD or YOLOv8n detector with minimal interface changes.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA-enabled GPU for faster training

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fire-detection-people-counting.git
cd fire-detection-people-counting
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Fire Detection — Real-time Webcam Inference

Requires a trained model saved as `InceptionV3.h5` in the root directory (see [Training](#-training) below).

```bash
python fire_detection/inference.py
```

- Press `q` to quit.
- When fire is detected, the frame is converted to grayscale as a visual indicator and the confidence score is printed to the console.

---

### People Counting

**From webcam:**
```bash
python people_counting/detector.py --camera true
```

**From a video file:**
```bash
python people_counting/detector.py --video path/to/video.mp4
```

**From an image:**
```bash
python people_counting/detector.py --image path/to/image.jpg --output path/to/output.jpg
```

**Save output video:**
```bash
python people_counting/detector.py --video path/to/video.mp4 --output path/to/result.avi
```

| Flag | Description |
|---|---|
| `-c / --camera` | Use live webcam feed (`true` / `false`) |
| `-v / --video` | Path to input video file |
| `-i / --image` | Path to input image |
| `-o / --output` | Path to save output video/image |

---

## 🏋️ Training

The fire detection model is trained in Google Colab. Open the notebook:

```
fire_detection/train.ipynb
```

**Steps the notebook performs:**
1. Downloads the Fire-Smoke Dataset from GitHub releases.
2. Removes the Smoke class, leaving a binary Fire / No-Fire problem.
3. Applies data augmentation and sets up `ImageDataGenerator` pipelines.
4. Builds the InceptionV3 transfer learning model.
5. Phase 1 training: custom head only (up to 20 epochs, early stopping at val_loss ≤ 0.11).
6. Phase 2 fine-tuning: unfreezes top layers, retrains with SGD.
7. Plots training/validation accuracy and loss curves.
8. Saves the model as `InceptionV3.h5`.

> **Note:** The notebook is designed to run on Google Colab (free GPU). The dataset download and unzip cells at the top handle all setup automatically.

---

## 📦 Dependencies

```
tensorflow>=2.10.0
keras
opencv-python>=4.5.0
imutils>=0.5.4
numpy>=1.21.0
Pillow>=9.0.0
matplotlib>=3.5.0
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 🔬 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~94% (Phase 2) |
| Validation Accuracy | ~91% (Phase 2) |
| Early Stop Criterion | val_loss ≤ 0.11 AND train_loss ≤ 0.11 |
| People Detection | Real-time, ~15–20 FPS on desktop CPU |

> Full training curves and sample outputs are documented in [`docs/results.md`](docs/results.md).

---

## 🔭 Potential Improvements

- Replace HOG+SVM with **YOLOv8n** for significantly improved people detection accuracy and speed on GPU.
- Add **SMTP / Twilio SMS alerting** when fire is detected to notify emergency contacts automatically.
- Integrate a **Flask/FastAPI web dashboard** to visualise camera feeds and counts in real time.
- Export the InceptionV3 model to **TensorFlow Lite** for more efficient deployment on Raspberry Pi.
- Add **multi-camera support** for building-wide coverage.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Alen Sebastian Veliyathuparamban**  
MSc Data and Computational Science — University College Dublin  
B.Tech Computer Science & Engineering (Robotics) — VIT  
[LinkedIn](https://www.linkedin.com/in/alen-sebastian-veliyathuparamban-880748201/) · [GitHub](https://github.com/ALEN2002-py)
