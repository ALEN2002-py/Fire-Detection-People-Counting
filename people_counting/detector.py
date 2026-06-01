"""
people_counting/detector.py
----------------------------
Real-time people detection and counting using the HOG + LinearSVM descriptor
built into OpenCV. Supports three input modes: webcam, video file, and image.

Usage examples:
    # Webcam
    python people_counting/detector.py --camera true

    # Video file
    python people_counting/detector.py --video path/to/video.mp4

    # Single image (saves annotated output)
    python people_counting/detector.py --image path/to/image.jpg --output result.jpg

    # Save output video
    python people_counting/detector.py --video input.mp4 --output output.avi

Author: Alen Mathew
"""

import argparse
import sys
import cv2
import imutils
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────

# HOG sliding-window parameters — tune these for speed vs. accuracy trade-off
HOG_WIN_STRIDE = (4, 4)   # Step size for the sliding window (pixels). Smaller = slower but more thorough.
HOG_PADDING = (8, 8)      # Padding added to each side of the detection window.
HOG_SCALE = 1.03          # Scale factor for the image pyramid. Values close to 1.0 = more scales = slower.
MAX_FRAME_WIDTH = 800     # Frames are resized to this max width before detection (preserves aspect ratio).


# ── Core Detection ─────────────────────────────────────────────────────────────

def detect_people(frame: np.ndarray, hog: cv2.HOGDescriptor) -> np.ndarray:
    """
    Run HOG+SVM people detection on a single frame and annotate it.

    For each detected person the function draws:
        - A green bounding rectangle.
        - A red label with the person index.
    A blue status bar at the top shows the current total count.

    Args:
        frame:  BGR image/frame (already resized to target width).
        hog:    Initialised cv2.HOGDescriptor with the default people detector loaded.

    Returns:
        Annotated BGR frame with bounding boxes and count overlay.
    """
    bounding_boxes, _ = hog.detectMultiScale(
        frame,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE
    )

    person_count = len(bounding_boxes)

    for idx, (x, y, w, h) in enumerate(bounding_boxes, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"person {idx}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Status overlay
    cv2.putText(frame, "Status: Detecting", (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Total Persons: {person_count}", (20, 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)

    return frame


# ── Input Mode Handlers ───────────────────────────────────────────────────────

def detect_from_video(video_path: str, writer, hog: cv2.HOGDescriptor) -> None:
    """
    Run people detection on a video file, frame by frame.

    Args:
        video_path: Full path to the input video file.
        writer:     cv2.VideoWriter instance (or None to skip saving).
        hog:        Initialised HOGDescriptor.
    """
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()

    if not success:
        print(f"[ERROR] Could not open video: {video_path}")
        print("[ERROR] Please provide the full, valid path to a video file.")
        sys.exit(1)

    print("[INFO] Processing video — press 'q' to quit.")

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        frame = imutils.resize(frame, width=min(MAX_FRAME_WIDTH, frame.shape[1]))
        frame = detect_people(frame, hog)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("People Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


def detect_from_camera(writer, hog: cv2.HOGDescriptor) -> None:
    """
    Run live people detection from the default webcam (device index 0).

    Args:
        writer: cv2.VideoWriter instance (or None to skip saving).
        hog:    Initialised HOGDescriptor.
    """
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        sys.exit(1)

    print("[INFO] Opening webcam — press 'q' to quit.")

    while True:
        success, frame = video.read()
        if not success:
            print("[ERROR] Failed to read from webcam.")
            break

        frame = imutils.resize(frame, width=min(MAX_FRAME_WIDTH, frame.shape[1]))
        frame = detect_people(frame, hog)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("People Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


def detect_from_image(image_path: str, output_path: str | None, hog: cv2.HOGDescriptor) -> None:
    """
    Run people detection on a single image and optionally save the result.

    Args:
        image_path:  Full path to the input image.
        output_path: Path to save the annotated output image (or None to skip).
        hog:         Initialised HOGDescriptor.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    image = imutils.resize(image, width=min(MAX_FRAME_WIDTH, image.shape[1]))
    result = detect_people(image, hog)

    if output_path is not None:
        cv2.imwrite(output_path, result)
        print(f"[INFO] Annotated image saved to: {output_path}")

    cv2.imshow("People Counting", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Dispatcher ────────────────────────────────────────────────────────────────

def run_detector(args: dict, hog: cv2.HOGDescriptor) -> None:
    """
    Route to the appropriate detection mode based on parsed CLI arguments.

    Priority order:
        1. Camera (--camera true)
        2. Video file (--video)
        3. Image file (--image)

    Args:
        args: Parsed argument dictionary from argsParser().
        hog:  Initialised HOGDescriptor.
    """
    use_camera = str(args["camera"]).lower() == "true"
    video_path = args["video"]
    image_path = args["image"]
    output_path = args["output"]

    # Initialise video writer only for camera/video modes with an output path
    writer = None
    if output_path is not None and image_path is None:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps=10,
            frameSize=(MAX_FRAME_WIDTH, 600)
        )

    if use_camera:
        print("[INFO] Starting webcam detection.")
        detect_from_camera(writer, hog)
    elif video_path is not None:
        print(f"[INFO] Processing video: {video_path}")
        detect_from_video(video_path, writer, hog)
    elif image_path is not None:
        print(f"[INFO] Processing image: {image_path}")
        detect_from_image(image_path, output_path, hog)
    else:
        print("[ERROR] No input mode specified. Use --camera, --video, or --image.")
        print("        Run with --help for usage details.")
        sys.exit(1)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> dict:
    """
    Parse command-line arguments.

    Returns:
        Dictionary of argument key-value pairs.
    """
    parser = argparse.ArgumentParser(
        description="People Detection & Counting using HOG + SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detector.py --camera true
  python detector.py --video path/to/video.mp4
  python detector.py --image path/to/image.jpg --output result.jpg
  python detector.py --video input.mp4 --output output.avi
        """
    )
    parser.add_argument("-v", "--video",  default=None,  help="Path to input video file.")
    parser.add_argument("-i", "--image",  default=None,  help="Path to input image file.")
    parser.add_argument("-c", "--camera", default=False, help="Set to 'true' to use the webcam.")
    parser.add_argument("-o", "--output", default=None,  help="Path to save the output video or image.")

    return vars(parser.parse_args())


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialise the HOG descriptor with OpenCV's default pre-trained people detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = parse_args()
    run_detector(args, hog)
