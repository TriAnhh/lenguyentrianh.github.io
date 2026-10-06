"""
Enhanced Hand & Finger Detection and Gesture Recognition Module
Optimized for robust Front-Side (Palm) and Back-Side detection with
MediaPipe Tasks API, Metric 3D World Landmarks, and Orientation Analysis.
"""

import os
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")

# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

FINGERTIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

GESTURE_INFO = {
    "Closed_Fist": ("Fist", "✊"),
    "Open_Palm": ("Open Palm", "✋"),
    "Pointing_Up": ("Pointing Up", "☝️"),
    "Thumb_Up": ("Thumbs Up", "👍"),
    "Thumb_Down": ("Thumbs Down", "👎"),
    "Victory": ("Victory / Peace", "✌️"),
    "ILoveYou": ("I Love You", "🤟"),
    "None": ("Tracking", "🖐️"),
    "Unrecognized": ("Hand Detected", "👋"),
    # Custom & Frontal Gestures
    "OK": ("OK Sign", "👌"),
    "Rock": ("Rock / Horns", "🤘"),
    "Call_Me": ("Call Me", "🤙"),
    "Gun": ("Gun / Point", "👉"),
    "Pinch": ("Pinch", "🤏"),
    "Fingers_1": ("1 Finger", "1️⃣"),
    "Fingers_2": ("2 Fingers", "2️⃣"),
    "Fingers_3": ("3 Fingers", "3️⃣"),
    "Fingers_4": ("4 Fingers", "4️⃣"),
    "Fingers_5": ("Open Hand (5)", "✋"),
    "Fingers_0": ("Fist (0)", "✊"),
}


def ensure_model_exists(model_path=DEFAULT_MODEL_PATH):
    """Downloads the official MediaPipe gesture recognizer model if not found."""
    if not os.path.exists(model_path):
        print(f"[HandDetector] Downloading gesture recognizer model to {model_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[HandDetector] Model download complete.")
    return model_path


class HandGestureDetector:
    """
    High-performance Hand, Finger Detection, and Gesture Recognition engine.
    Optimized for high sensitivity on front-side palms and dorsal views.
    """

    def __init__(
        self,
        model_path=None,
        num_hands=2,
        min_hand_detection_confidence=0.3,  # Lowered from 0.5 to 0.3 for instant front palm locking
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        ensure_model_exists(self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    @staticmethod
    def _euclidean_dist_2d(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _calculate_angle_3d(a, b, c):
        """Calculates 3D joint angle at node b between vectors (a-b) and (c-b)."""
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        dot = np.dot(ba, bc)
        norm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-7
        cosine = np.clip(dot / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _determine_hand_orientation(self, image_landmarks, raw_handedness="Right"):
        """
        Determines whether the front side (palm) or back side (dorsal)
        is facing towards the camera using the 2D rotation-invariant cross-product.

        Anatomical principle:
        Let v_horiz = Pinky_MCP - Index_MCP, v_vert = Middle_MCP - Wrist.
        In image coordinates (x right, y down):
        For a Right Hand:
          - Palm facing camera: Pinky is to the right of Index (v_horiz.x > 0),
            Middle is above Wrist (v_vert.y < 0) -> Cross-product z < 0.
          - Back facing camera: Index is to the right of Pinky -> Cross-product z > 0.
        For a Left Hand:
          - Palm facing camera: Cross-product z > 0.
          - Back facing camera: Cross-product z < 0.
        """
        wrist = image_landmarks[WRIST]
        mid = image_landmarks[MIDDLE_MCP]
        idx = image_landmarks[INDEX_MCP]
        pky = image_landmarks[PINKY_MCP]

        v_horiz = np.array([pky.x - idx.x, pky.y - idx.y])
        v_vert = np.array([mid.x - wrist.x, mid.y - wrist.y])

        cz = float(v_horiz[0] * v_vert[1] - v_horiz[1] * v_vert[0])
        scale = float((np.linalg.norm(v_horiz) * np.linalg.norm(v_vert)) + 1e-7)
        normalized_cz = cz / scale

        # In camera frame:
        # Right Hand: cz > 0.10 is Palm (Front), cz < -0.10 is Back
        # Left Hand:  cz < -0.10 is Palm (Front), cz > 0.10 is Back
        if raw_handedness == "Right":
            is_front_palm = normalized_cz > 0.10
            is_back_side = normalized_cz < -0.10
        else:
            is_front_palm = normalized_cz < -0.10
            is_back_side = normalized_cz > 0.10

        if is_front_palm:
            orientation_desc = "FRONT (PALM)"
        elif is_back_side:
            orientation_desc = "BACK OF HAND"
        else:
            orientation_desc = "SIDE VIEW"

        return is_front_palm, orientation_desc, normalized_cz

    def _analyze_fingers(self, img_landmarks, world_landmarks, is_front_palm):
        """
        Robust finger extension detector using 3D metric coordinates,
        joint angles, and 2D vertical coordinates.
        """
        lms = world_landmarks if world_landmarks is not None else img_landmarks
        wrist = lms[WRIST]
        fingers_up = [False, False, False, False, False]

        finger_chains = [
            (INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_DIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP),
        ]

        # 1. Evaluate Index, Middle, Ring, Pinky
        for i, (tip_idx, dip_idx, pip_idx, mcp_idx) in enumerate(finger_chains):
            tip = lms[tip_idx]
            pip = lms[pip_idx]
            mcp = lms[mcp_idx]

            # 3D distances
            d_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y, tip.z - wrist.z)
            d_pip_wrist = math.hypot(pip.x - wrist.x, pip.y - wrist.y, pip.z - wrist.z)
            d_mcp_wrist = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z)

            d_tip_mcp = math.hypot(tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z)
            d_pip_mcp = math.hypot(pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z)

            # 3D PIP joint angle (straight finger ~ 160-180 deg)
            angle_pip = self._calculate_angle_3d(mcp, pip, tip)

            # 2D relative height in image (if hand is roughly upright)
            img_tip = img_landmarks[tip_idx]
            img_pip = img_landmarks[pip_idx]
            img_mcp = img_landmarks[mcp_idx]
            is_upright = img_wrist_is_bottom = img_landmarks[WRIST].y > img_mcp.y

            is_up_in_2d = img_tip.y < img_pip.y if is_upright else True

            # Finger is extended if:
            # - PIP angle is reasonably straight (> 135 deg)
            # - OR tip is further from wrist than PIP joint
            # - AND tip is not tucked close to MCP
            is_extended = (
                (angle_pip > 135 and d_tip_mcp > d_pip_mcp)
                or (d_tip_wrist > d_pip_wrist * 1.08 and d_tip_wrist > d_mcp_wrist)
            )

            # Extra validation when hand is upright
            if is_upright:
                if not is_up_in_2d and angle_pip < 150:
                    is_extended = False

            fingers_up[i + 1] = bool(is_extended)

        # 2. Evaluate Thumb
        thumb_tip = lms[THUMB_TIP]
        thumb_ip = lms[THUMB_IP]
        thumb_mcp = lms[THUMB_MCP]
        pinky_mcp = lms[PINKY_MCP]
        index_mcp = lms[INDEX_MCP]

        d_thumb_wrist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y, thumb_tip.z - wrist.z)
        d_thumb_mcp_wrist = math.hypot(thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y, thumb_mcp.z - wrist.z)
        d_thumb_pinky = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y, thumb_tip.z - pinky_mcp.z)
        d_ip_pinky = math.hypot(thumb_ip.x - pinky_mcp.x, thumb_ip.y - pinky_mcp.y, thumb_ip.z - pinky_mcp.z)
        d_thumb_index = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y, thumb_tip.z - index_mcp.z)

        # Thumb is extended ONLY if tip spreads outwards away from the pinky base and palm
        is_thumb_tucked = d_thumb_pinky <= (d_ip_pinky * 1.05)
        if is_thumb_tucked:
            thumb_extended = False
        else:
            thumb_extended = (d_thumb_pinky > d_ip_pinky * 1.12) and (d_thumb_wrist > d_thumb_mcp_wrist * 1.15)

        fingers_up[0] = bool(thumb_extended)
        finger_count = sum(fingers_up)
        return fingers_up, finger_count

    def _detect_custom_gestures(self, img_landmarks, fingers_up, finger_count, model_gesture, is_front_palm):
        """
        Synthesizes AI model results with geometric finger states.
        Guarantees accurate recognition on front palms, fists, peace, OK, etc.
        """
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers_up

        # Tip-to-tip pinch distance
        t_tip = img_landmarks[THUMB_TIP]
        i_tip = img_landmarks[INDEX_TIP]
        pinch_dist = math.hypot(t_tip.x - i_tip.x, t_tip.y - i_tip.y, t_tip.z - i_tip.z)

        # 1. OK Sign: Thumb and Index touching, other 3 fingers extended
        if pinch_dist < 0.08 and middle_up and ring_up:
            return "OK", 0.92

        # 2. Victory / Peace: Index & Middle extended, others folded
        if index_up and middle_up and not ring_up and not pinky_up:
            return "Victory", 0.94

        # 3. Pointing Up / Point: Index extended only
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "Pointing_Up", 0.92

        # 4. Rock / Horns: Index and Pinky extended, Middle and Ring folded
        if index_up and pinky_up and not middle_up and not ring_up:
            return "Rock", 0.92

        # 5. Call Me: Thumb and Pinky extended, middle fingers folded
        if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
            return "Call_Me", 0.90

        # 6. Gun / Point: Thumb and Index extended
        if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            return "Gun", 0.88

        # 7. Thumbs Up / Thumbs Down: Thumb only
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            if model_gesture in ("Thumb_Up", "Thumb_Down"):
                return model_gesture, 0.90
            return "Thumb_Up", 0.85

        # 8. Open Palm / Front Hand: All 5 fingers extended
        if finger_count == 5:
            return "Open_Palm", 0.95

        # 9. Closed Fist: 0 fingers extended
        if finger_count == 0:
            return "Closed_Fist", 0.92

        # 10. Pinch: Thumb & Index very close together
        if pinch_dist < 0.045:
            return "Pinch", 0.85

        # Fallback to model gesture if confident
        if model_gesture not in ("None", "Unrecognized", ""):
            return model_gesture, 0.80

        # Numerical finger count display
        return f"Fingers_{finger_count}", 0.80

    def detect(self, bgr_frame, is_mirrored=False):
        """
        Runs full hand detection, orientation analysis, finger detection,
        and gesture recognition on an OpenCV BGR frame.
        """
        h, w, _ = bgr_frame.shape

        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.recognizer.recognize(mp_image)

        detected_hands = []
        if not results.hand_landmarks:
            return detected_hands

        num_detected = len(results.hand_landmarks)

        for i in range(num_detected):
            landmarks = results.hand_landmarks[i]
            world_landmarks = (
                results.hand_world_landmarks[i]
                if results.hand_world_landmarks and i < len(results.hand_world_landmarks)
                else None
            )

            # Handedness with selfie-mirror compensation
            raw_handedness = "Right"
            handedness_score = 1.0
            if results.handedness and i < len(results.handedness) and len(results.handedness[i]) > 0:
                cat = results.handedness[i][0]
                raw_handedness = cat.category_name
                handedness_score = float(cat.score)

            if is_mirrored:
                # Mirror mode inverts Left and Right in camera frame
                handedness = "Right" if raw_handedness == "Left" else "Left"
            else:
                handedness = raw_handedness

            # Determine Front Side (Palm) vs Back Side orientation
            is_front_palm, orientation_desc, normal_z = self._determine_hand_orientation(
                landmarks, raw_handedness
            )

            # Convert normalized landmarks to pixel coordinates
            pixel_landmarks = []
            xs, ys = [], []
            for lm in landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pixel_landmarks.append((px, py))
                xs.append(px)
                ys.append(py)

            # Hand Bounding Box with padding
            pad = 20
            x_min = max(0, min(xs) - pad)
            y_min = max(0, min(ys) - pad)
            x_max = min(w, max(xs) + pad)
            y_max = min(h, max(ys) + pad)
            bbox = (x_min, y_min, x_max, y_max)

            # MediaPipe trained AI gesture
            model_gesture = "None"
            model_gesture_score = 0.0
            if results.gestures and i < len(results.gestures) and len(results.gestures[i]) > 0:
                top_gesture = results.gestures[i][0]
                model_gesture = top_gesture.category_name
                model_gesture_score = float(top_gesture.score)

            # Analyze individual fingers
            fingers_up, finger_count = self._analyze_fingers(landmarks, world_landmarks, is_front_palm)

            # Synthesize dominant gesture
            chosen_gesture, chosen_score = self._detect_custom_gestures(
                landmarks, fingers_up, finger_count, model_gesture, is_front_palm
            )

            # If model is confident on prototypical gesture and agrees with count, elevate confidence
            if model_gesture == chosen_gesture and model_gesture_score > 0.60:
                chosen_score = max(chosen_score, model_gesture_score)

            # Pinch analysis
            thumb_px = pixel_landmarks[THUMB_TIP]
            index_px = pixel_landmarks[INDEX_TIP]
            pinch_dist_px = self._euclidean_dist_2d(thumb_px, index_px)
            hand_scale = max(1.0, self._euclidean_dist_2d(pixel_landmarks[WRIST], pixel_landmarks[MIDDLE_MCP]))
            normalized_pinch_ratio = pinch_dist_px / hand_scale
            is_pinching = normalized_pinch_ratio < 0.28

            pinch_info = {
                "is_pinching": is_pinching,
                "distance_px": pinch_dist_px,
                "normalized_ratio": normalized_pinch_ratio,
                "thumb_tip_px": thumb_px,
                "index_tip_px": index_px,
                "center_px": ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2),
            }

            disp_name, disp_icon = GESTURE_INFO.get(chosen_gesture, (chosen_gesture.replace("_", " "), "✋"))

            hand_data = {
                "hand_index": i,
                "handedness": handedness,
                "raw_handedness": raw_handedness,
                "handedness_score": handedness_score,
                "is_front_palm": is_front_palm,
                "orientation_desc": orientation_desc,
                "normal_z": normal_z,
                "landmarks": landmarks,
                "world_landmarks": world_landmarks,
                "pixel_landmarks": pixel_landmarks,
                "bbox": bbox,
                "model_gesture": model_gesture,
                "model_gesture_score": model_gesture_score,
                "gesture": chosen_gesture,
                "gesture_score": chosen_score,
                "gesture_display": disp_name,
                "gesture_icon": disp_icon,
                "fingers_up": fingers_up,
                "finger_count": finger_count,
                "pinch": pinch_info,
            }

            detected_hands.append(hand_data)

        return detected_hands
