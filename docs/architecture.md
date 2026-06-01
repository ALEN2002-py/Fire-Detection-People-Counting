# System Architecture

## Overview

The system consists of two independent computer vision modules that can be run separately or combined into a unified pipeline.

```
┌─────────────────────────────────────────────────────┐
│                  Camera / Video Input                │
└────────────────────────┬────────────────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │       Frame Capture         │
           │      (cv2.VideoCapture)     │
           └──────┬───────────┬─────────┘
                  │           │
    ┌─────────────▼─┐   ┌─────▼──────────────┐
    │ Fire Detection │   │  People Counting    │
    │  (InceptionV3) │   │   (HOG + SVM)       │
    └───────┬────────┘   └──────┬─────────────┘
            │                   │
    ┌───────▼────────┐  ┌───────▼──────────────┐
    │ Fire/No-Fire   │  │ Bounding Boxes +      │
    │ + Confidence   │  │ Person Count          │
    └───────┬────────┘  └──────┬───────────────┘
            │                   │
            └──────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Annotated Frame │
              │  (Display/Save)  │
              └─────────────────┘
```

---

## Module 1 — Fire Detection

### Model Architecture

| Layer | Type | Output Shape | Notes |
|---|---|---|---|
| Input | Input | (224, 224, 3) | RGB normalised to [0,1] |
| InceptionV3 base | CNN backbone | (5, 5, 2048) | ImageNet weights, frozen in Phase 1 |
| GlobalAveragePooling2D | Pooling | (2048,) | Reduces spatial dims to single vector |
| Dense | Fully connected | (2048,) | ReLU activation |
| Dropout | Regularisation | (2048,) | p=0.25, prevents overfitting |
| Dense | Fully connected | (1024,) | ReLU activation |
| Dropout | Regularisation | (1024,) | p=0.20 |
| Dense | Output | (2,) | Softmax — [Fire, No-Fire] |

### Training Protocol

**Phase 1 — Feature extraction**
- Freeze all InceptionV3 base layers.
- Train only the custom classification head.
- Optimiser: RMSprop (default lr).
- Epochs: up to 20, with early stopping when both train_loss and val_loss ≤ 0.11.

**Phase 2 — Fine-tuning**
- Unfreeze InceptionV3 layers from index 249 onward.
- Use a much lower learning rate (SGD, lr=0.0001, momentum=0.9) to prevent destroying pre-trained weights.
- Epochs: up to 10.

### Why InceptionV3?

- Strong ImageNet pre-training makes it an excellent feature extractor for visual domains like fire.
- Factorised convolutions (the "Inception" trick) give a good accuracy-to-parameter trade-off vs. VGG or ResNet.
- Small enough to run inference on Raspberry Pi with TFLite conversion.

---

## Module 2 — People Counting

### Method: Histogram of Oriented Gradients (HOG) + Linear SVM

HOG captures the distribution of gradient orientations in local image patches — highly effective for describing the shape and edge structure of human silhouettes.

**Detection pipeline:**
1. Build an image pyramid (scale factor 1.03 per level).
2. Slide a 64×128 window across each pyramid level with a stride of 4×4 pixels.
3. Compute the HOG descriptor for each window.
4. Pass the descriptor through a pre-trained Linear SVM (`cv2.HOGDescriptor_getDefaultPeopleDetector()`).
5. Collect positive detections; draw bounding boxes.

### Parameter Choices

| Parameter | Value | Rationale |
|---|---|---|
| winStride | (4, 4) | Small stride for thorough coverage at the cost of speed |
| padding | (8, 8) | Adds context around each detection window |
| scale | 1.03 | Fine-grained pyramid — detects a wider range of person sizes |
| maxWidth | 800px | Caps frame width for consistent speed on low-power hardware |

### Trade-offs

| Aspect | HOG+SVM | Alternative (e.g. YOLOv8n) |
|---|---|---|
| Accuracy | Moderate | High |
| Speed on CPU | Good | Good (with ONNX) |
| GPU required | No | Optional |
| Raspberry Pi support | Native OpenCV | Requires TFLite/ONNX export |
| Crowd scenes | Poor (occlusion) | Better |

For production deployment where accuracy is critical, replacing HOG+SVM with a lightweight neural detector (MobileNet-SSD or YOLOv8n) would be the recommended next step.

---

## Edge Deployment — Raspberry Pi

The system was designed with Raspberry Pi (3B+ or 4) deployment in mind.

**Recommended setup:**
- Convert InceptionV3 to TFLite (`tf.lite.TFLiteConverter`) for 3–5× faster inference on ARM.
- Use a Raspberry Pi Camera Module V2 instead of USB webcam for lower latency.
- Use `picamera2` library instead of `cv2.VideoCapture` on Pi OS Bookworm.
- Set the `scale` parameter in HOG detection to 1.05+ to reduce pyramid levels and improve FPS.
