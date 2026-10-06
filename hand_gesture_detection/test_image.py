"""
Standalone Test Script for Hand & Gesture Recognition
Runs detection on a static image and outputs the annotated result to disk.
"""

import os
import sys
import argparse
import urllib.request
import cv2

from detector import HandGestureDetector
from visualizer import (
    draw_hand_skeleton,
    draw_pinch_overlay,
    draw_hand_badge,
    draw_hud,
)

SAMPLE_IMAGE_URL = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg"
DEFAULT_TEST_IMAGE = os.path.join(os.path.dirname(__file__), "sample_hand.jpg")


def fetch_sample_if_needed(image_path):
    if not os.path.exists(image_path):
        print(f"[Test] Fetching sample hand image to {image_path}...")
        try:
            # Try fetching sample image from GitHub
            req = urllib.request.Request(
                SAMPLE_IMAGE_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response, open(image_path, "wb") as out_file:
                out_file.write(response.read())
            print("[Test] Sample image downloaded.")
        except Exception as e:
            print(f"[Test] Note: Could not download sample image ({e}).")
            # Generate a blank image canvas for synthetic testing if needed
            return False
    return True


def run_test_on_image(image_path=None, output_path=None):
    if image_path is None:
        image_path = DEFAULT_TEST_IMAGE
        fetch_sample_if_needed(image_path)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "output_test_result.png")

    if not os.path.exists(image_path):
        print(f"[Test] Image not found at {image_path}.")
        return False

    print(f"[Test] Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Test] Failed to decode image: {image_path}")
        return False

    detector = HandGestureDetector(num_hands=2)
    print("[Test] Running hand detection and gesture recognition...")
    detected_hands = detector.detect(frame)

    print(f"\n[Test Results] Detected {len(detected_hands)} hand(s):")
    for i, hand in enumerate(detected_hands):
        print(f"  Hand #{i + 1}:")
        print(f"    - Handedness: {hand['handedness']} ({hand['handedness_score']:.2f})")
        print(f"    - Orientation: {hand.get('orientation_desc')} (Front Palm: {hand.get('is_front_palm')})")
        print(f"    - Gesture: {hand['gesture_display']} [{hand['gesture']}] (Confidence: {hand['gesture_score']:.2f})")
        print(f"    - Fingers Extended: {hand['finger_count']}/5 -> {hand['fingers_up']}")
        print(f"    - Bounding Box: {hand['bbox']}")
        print(f"    - Pinch Status: {hand['pinch']['is_pinching']} (Ratio: {hand['pinch']['normalized_ratio']:.3f})")

    # Render visualizations
    annotated = frame.copy()
    for hand in detected_hands:
        draw_hand_skeleton(annotated, hand)
        draw_pinch_overlay(annotated, hand)
        draw_hand_badge(annotated, hand)

    draw_hud(annotated, detected_hands, fps=30.0, show_help=False)

    cv2.imwrite(output_path, annotated)
    print(f"\n[Test] Annotated image saved to: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output")
    args = parser.parse_args()

    run_test_on_image(args.image, args.output)
