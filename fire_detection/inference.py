"""
fire_detection/inference.py
---------------------------
Real-time fire detection from a webcam using a fine-tuned InceptionV3 model.

Usage:
    python fire_detection/inference.py

Controls:
    Press 'q' to quit.

Requirements:
    - Trained model file: InceptionV3.h5 (in project root or same directory)
    - See train.ipynb for training instructions.

Author: Alen Sebastian
"""

import sys
import cv2
import numpy as np
from PIL import Image

# TensorFlow is imported lazily (inside load_model / run_inference) so that
# the preprocessing and annotation functions can be imported and tested in
# environments where TensorFlow is not installed (e.g. CI runners).
keras_image = None  # populated by _import_tensorflow() at runtime


def _import_tensorflow():
    """Import TensorFlow and set the module-level keras_image alias."""
    global keras_image
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image as _keras_image
        keras_image = _keras_image
        return tf
    except ImportError:
        print("[ERROR] TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = "InceptionV3.h5"   # Path to the saved Keras model
INPUT_SIZE = (224, 224)         # Must match the size used during training
FIRE_CLASS_INDEX = 0            # Index 0 = Fire, Index 1 = No-Fire (per training class ordering)
CONFIDENCE_THRESHOLD = 0.70     # Minimum confidence to trigger a fire alert


# ── Helper Functions ──────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Load the trained Keras model from disk.

    Args:
        model_path: Path to the .h5 model file.

    Returns:
        Loaded tf.keras.Model ready for inference.

    Raises:
        SystemExit: If the model file is not found.
    """
    try:
        tf = _import_tensorflow()
        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Model loaded from: {model_path}")
        return model
    except OSError:
        print(f"[ERROR] Model file not found: {model_path}")
        print("[ERROR] Please train the model first using fire_detection/train.ipynb")
        sys.exit(1)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a single BGR frame from OpenCV for model inference.

    Steps:
        1. Convert BGR (OpenCV default) → RGB (PIL/Keras convention)
        2. Resize to the expected input size (224×224)
        3. Expand dims to create a batch of 1: shape (1, 224, 224, 3)
        4. Normalise pixel values from [0, 255] → [0.0, 1.0]

    Args:
        frame: Raw BGR frame from cv2.VideoCapture.

    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3).
    """
    pil_image = Image.fromarray(frame, "RGB")
    pil_image = pil_image.resize(INPUT_SIZE)
    img_array = np.array(pil_image, dtype=np.float32)  # equivalent to keras img_to_array, no TF needed
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def annotate_frame(frame: np.ndarray, is_fire: bool, confidence: float) -> np.ndarray:
    """
    Overlay the detection result onto the frame.

    If fire is detected:
        - Converts the frame to grayscale as a visual indicator.
        - Overlays a red "FIRE DETECTED" label with confidence score.
    Otherwise:
        - Overlays a green "No Fire" label.

    Args:
        frame:      Raw BGR frame.
        is_fire:    True if fire was detected.
        confidence: Model confidence score for the prediction.

    Returns:
        Annotated BGR frame.
    """
    if is_fire:
        # Grayscale conversion is a deliberate visual cue — easy to notice
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to 3-ch for text overlay

        label = f"FIRE DETECTED ({confidence:.2%})"
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
    else:
        label = f"No Fire ({confidence:.2%})"
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 0), 2)

    return frame


# ── Main Inference Loop ────────────────────────────────────────────────────────

def run_inference(model) -> None:
    """
    Open the default webcam and run frame-by-frame fire detection.

    The loop:
        1. Captures a frame.
        2. Preprocesses it.
        3. Runs model inference.
        4. Annotates the frame with the result.
        5. Displays the annotated frame.
        6. Exits on 'q' keypress.

    Args:
        model: Loaded Keras model.
    """
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        sys.exit(1)

    print("[INFO] Starting real-time fire detection. Press 'q' to quit.")

    while True:
        success, frame = video.read()
        if not success:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Preprocess and predict
        input_tensor = preprocess_frame(frame)
        probabilities = model.predict(input_tensor, verbose=0)[0]

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        is_fire = (predicted_class == FIRE_CLASS_INDEX) and (confidence >= CONFIDENCE_THRESHOLD)

        if is_fire:
            print(f"[ALERT] Fire detected! Confidence: {confidence:.2%}")

        # Annotate and display
        annotated = annotate_frame(frame, is_fire, confidence)
        cv2.imshow("Fire Detection System", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quitting.")
            break

    video.release()
    cv2.destroyAllWindows()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    run_inference(model)
