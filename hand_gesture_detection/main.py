"""
Hand & Finger Detection and Gesture Recognition Application
Real-time tracking via webcam with Front-Side Palm & Dorsal detection optimizations.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import cv2

from detector import HandGestureDetector
from visualizer import (
    draw_hand_skeleton,
    draw_pinch_overlay,
    draw_hand_badge,
    draw_hud,
    apply_clahe_contrast_enhancement,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Hand & Finger Detection + Gesture Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width resolution (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height resolution (default: 720)")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect (default: 2)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror / selfie view")
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.3,  # Lowered default for responsive front-side palm detection
        help="Min detection confidence (0.0-1.0, default: 0.3)",
    )
    return parser.parse_args()


def init_camera(camera_idx, width, height):
    """Initializes webcam with hardware MJPEG decoding and buffer flushing for zero lag."""
    print(f"[Camera] Opening camera device index {camera_idx}...")
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"[Camera] DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(camera_idx)

    if cap.isOpened():
        # Set MJPG compression for high FPS and low motion blur on Windows
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    args = parse_args()
    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    print("=" * 60)
    print("      HAND & FINGER DETECTION + GESTURE RECOGNITION      ")
    print("=" * 60)
    print("[1] Initializing MediaPipe AI gesture recognizer (Front & Back Hand Optimized)...")

    detector = HandGestureDetector(
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.detection_conf,
        min_hand_presence_confidence=args.detection_conf,
        min_tracking_confidence=args.detection_conf,
    )
    print(f"[1] Detector loaded successfully (Sensitivity threshold: {args.detection_conf}).")

    print(f"[2] Initializing camera index {args.camera}...")
    cap = init_camera(args.camera, args.width, args.height)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open webcam (index {args.camera}).")
        print("Tip: If you have an external webcam or another camera index, try running:")
        print("     py main.py --camera 1")
        print("Or run test_image.py to test on static images without a camera.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Resolution: {actual_w}x{actual_h}")

    # Display window setup
    window_name = "Hand & Gesture Recognition AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    # Runtime toggles
    show_skeleton = True
    show_box = True
    show_pinch = True
    show_hud_overlay = True
    contrast_boost = False
    mirror_mode = not args.no_mirror
    paused = False
    last_frame = None

    # FPS counter
    prev_time = time.time()
    fps = 0.0

    print("\n[Controls]")
    print("  [S]      Toggle Hand Skeleton & Fingertip Joints")
    print("  [B]      Toggle Hand Bounding Box & Floating Badge")
    print("  [C]      Toggle Contrast Boost (Enhances Palm in Flat/Washed Lighting)")
    print("  [P]      Toggle Pinch Visualizer")
    print("  [H]      Toggle Cyber HUD Overlay")
    print("  [M]      Toggle Mirror / Selfie Mode")
    print("  [SPACE]  Pause / Resume Feed")
    print("  [ENTER]  Save Snapshot to ./snapshots/")
    print("  [Q/ESC]  Quit")
    print("-" * 60)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Camera] Warning: Failed to read frame from camera.")
                    time.sleep(0.05)
                    continue

                if mirror_mode:
                    frame = cv2.flip(frame, 1)

                last_frame = frame.copy()
            else:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    time.sleep(0.05)
                    continue

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant_fps

            # Optional Contrast Boost (helpful for overexposed front palm)
            process_frame = frame
            if contrast_boost:
                process_frame = apply_clahe_contrast_enhancement(frame)

            # Detect Hands and Recognize Gestures
            detected_hands = detector.detect(process_frame, is_mirrored=mirror_mode)

            # Draw visual layers
            display_frame = frame.copy()
            if show_skeleton:
                for hand in detected_hands:
                    draw_hand_skeleton(display_frame, hand, draw_joints=True, draw_fingertips=True)

            if show_pinch:
                for hand in detected_hands:
                    draw_pinch_overlay(display_frame, hand)

            if show_box:
                for hand in detected_hands:
                    draw_hand_badge(display_frame, hand)

            if show_hud_overlay:
                draw_hud(display_frame, detected_hands, fps=fps, show_help=True)

            # Display Contrast Boost banner if active
            if contrast_boost:
                cv2.putText(
                    display_frame,
                    "[CONTRAST BOOST ACTIVE]",
                    (actual_w - 240, actual_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if paused:
                h, w, _ = display_frame.shape
                cv2.putText(
                    display_frame,
                    "[PAUSED - PRESS SPACE TO RESUME]",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display
            cv2.imshow(window_name, display_frame)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                print("\n[App] Exiting...")
                break
            elif key in (ord("s"), ord("S")):
                show_skeleton = not show_skeleton
                print(f"[Toggle] Skeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key in (ord("b"), ord("B")):
                show_box = not show_box
                print(f"[Toggle] Bounding Box: {'ON' if show_box else 'OFF'}")
            elif key in (ord("c"), ord("C")):
                contrast_boost = not contrast_boost
                print(f"[Toggle] Contrast Boost: {'ON' if contrast_boost else 'OFF'}")
            elif key in (ord("p"), ord("P")):
                show_pinch = not show_pinch
                print(f"[Toggle] Pinch Overlay: {'ON' if show_pinch else 'OFF'}")
            elif key in (ord("h"), ord("H")):
                show_hud_overlay = not show_hud_overlay
                print(f"[Toggle] HUD: {'ON' if show_hud_overlay else 'OFF'}")
            elif key in (ord("m"), ord("M")):
                mirror_mode = not mirror_mode
                print(f"[Toggle] Mirror Mode: {'ON' if mirror_mode else 'OFF'}")
            elif key == ord(" "):  # Space
                paused = not paused
                print(f"[Toggle] Paused: {paused}")
            elif key in (13, 10):  # Enter
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(snapshots_dir, f"snapshot_{timestamp}.png")
                cv2.imwrite(snap_path, display_frame)
                print(f"[Snapshot] Saved to {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Camera] Camera released and windows closed.")


if __name__ == "__main__":
    main()
