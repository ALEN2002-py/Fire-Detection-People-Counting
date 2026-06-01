# Results & Evaluation

## Fire Detection — InceptionV3

### Dataset

| Split | Classes | Approx. Samples |
|---|---|---|
| Train | Fire, No-Fire | ~1,800 |
| Validation | Fire, No-Fire | ~400 |

Source: [DeepQuestAI Fire-Smoke Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset)

The Smoke class was excluded to keep this a clean binary problem. Smoke-as-fire false positives are a known challenge and would require a separate multi-class model or a post-processing rule to handle correctly.

### Training Metrics

| Phase | Epochs Run | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|---|
| Phase 1 (head only) | ~15 | ~88% | ~86% | ~0.28 | ~0.32 |
| Phase 2 (fine-tune) | ~8  | ~94% | ~91% | ~0.10 | ~0.11 |

> Phase 2 training triggered the early stopping callback (both losses ≤ 0.11).

### Observations

- Transfer learning dramatically reduced the training time needed to reach good accuracy — the ImageNet features generalise well to the fire domain because fire detection relies on colour, texture, and shape features that overlap with general image recognition.
- The Dropout layers (0.25, 0.20) were important — without them, the model overfit the training set in Phase 1 within ~5 epochs.
- The confidence threshold in `inference.py` is set to 0.70. Lowering this increases recall (fewer missed fires) at the cost of more false alarms; this is the right trade-off for a safety application.

---

## People Counting — HOG+SVM

### Performance Characteristics

| Metric | Result |
|---|---|
| Approximate FPS (desktop CPU) | 15–20 FPS at 800px width |
| Approximate FPS (Raspberry Pi 4) | 4–6 FPS at 640px width |
| Detection quality | Good for uncrowded scenes (1–5 people) |
| Occlusion handling | Poor — HOG struggles when people overlap significantly |

### Known Limitations

1. **Crowded scenes:** When people are closely packed or occlude each other, HOG+SVM may merge detections or miss individuals entirely. A neural detector (YOLOv8, MobileNet-SSD) handles this significantly better.
2. **Partial bodies:** The detector struggles when only the upper or lower half of a person is visible (e.g. seated behind a desk).
3. **Low light:** HOG is gradient-based; in very low-light conditions gradients are noisy and detection quality degrades.

---

## Combined System

When both modules run together on a live camera feed, the typical use case is:

1. Fire module raises an alert (confidence > 0.70).
2. People counting module simultaneously reports the headcount.
3. Emergency services are informed of both the fire status and the number of people potentially trapped.

This provides substantially more actionable information than either module alone.
