"""
tests/test_inference.py
------------------------
Unit tests for the fire detection and people counting inference modules.

These tests mock the model and camera so they can run without a GPU,
trained model file, or connected camera.

Run with:
    python -m pytest tests/

Author: Alen Mathew
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path so we can import from fire_detection and people_counting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fire Detection Tests ───────────────────────────────────────────────────────

class TestPreprocessFrame:
    """Tests for the frame preprocessing pipeline in fire_detection/inference.py."""

    def test_output_shape(self):
        """preprocess_frame should return shape (1, 224, 224, 3)."""
        from fire_detection.inference import preprocess_frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(dummy_frame)
        assert result.shape == (1, 224, 224, 3), f"Expected (1,224,224,3), got {result.shape}"

    def test_pixel_normalisation(self):
        """Pixel values should be in [0.0, 1.0] after preprocessing."""
        from fire_detection.inference import preprocess_frame
        dummy_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = preprocess_frame(dummy_frame)
        assert result.max() <= 1.0, "Max pixel value exceeds 1.0 after normalisation"
        assert result.min() >= 0.0, "Min pixel value below 0.0 after normalisation"

    def test_handles_different_frame_sizes(self):
        """preprocess_frame should accept frames of any size and return fixed (1,224,224,3)."""
        from fire_detection.inference import preprocess_frame
        for h, w in [(240, 320), (720, 1280), (100, 100)]:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = preprocess_frame(frame)
            assert result.shape == (1, 224, 224, 3)


class TestAnnotateFrame:
    """Tests for fire_detection/inference.py annotate_frame function."""

    def _make_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_fire_returns_frame_same_shape(self):
        """annotate_frame should return a frame with the same shape."""
        from fire_detection.inference import annotate_frame
        frame = self._make_frame()
        result = annotate_frame(frame, is_fire=True, confidence=0.92)
        assert result.shape == (480, 640, 3)

    def test_no_fire_returns_frame_same_shape(self):
        """annotate_frame should return a frame with the same shape when no fire."""
        from fire_detection.inference import annotate_frame
        frame = self._make_frame()
        result = annotate_frame(frame, is_fire=False, confidence=0.20)
        assert result.shape == (480, 640, 3)

    def test_fire_converts_to_grayscale_appearance(self):
        """When fire is detected, the annotated frame should look grayscale (R==G==B per pixel)."""
        from fire_detection.inference import annotate_frame
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = annotate_frame(frame.copy(), is_fire=True, confidence=0.95)
        # After gray conversion and back, R==G==B for every pixel (ignoring text overlay area)
        # Check a safe non-text region
        region = result[100:200, 100:400]
        assert np.all(region[:, :, 0] == region[:, :, 1]), "R channel != G channel after grayscale conversion"
        assert np.all(region[:, :, 1] == region[:, :, 2]), "G channel != B channel after grayscale conversion"


# ── People Counting Tests ──────────────────────────────────────────────────────

class TestDetectPeople:
    """Tests for people_counting/detector.py detect_people function."""

    def test_returns_ndarray(self):
        """detect_people should return a numpy array."""
        import cv2
        from people_counting.detector import detect_people

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_people(frame, hog)
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self):
        """detect_people should return a frame with the same shape as input."""
        import cv2
        from people_counting.detector import detect_people

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_people(frame, hog)
        assert result.shape == (480, 640, 3)


# ── Argument Parsing Tests ─────────────────────────────────────────────────────

class TestArgsParser:
    """Tests for people_counting/detector.py argument parser."""

    def test_defaults(self, monkeypatch):
        """All arguments should default to None / False when not provided."""
        from people_counting.detector import parse_args
        monkeypatch.setattr(sys, "argv", ["detector.py"])
        args = parse_args()
        assert args["video"] is None
        assert args["image"] is None
        assert args["output"] is None
        assert str(args["camera"]).lower() in ("false", "none", "0")
