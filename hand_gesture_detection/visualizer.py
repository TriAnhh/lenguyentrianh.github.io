"""
Visualizer and HUD Rendering Module for Hand & Gesture Recognition
Draws cyber/tech-style overlays, skeleton lines, joint nodes, and status dashboards.
"""

import cv2
import numpy as np
from detector import (
    HAND_CONNECTIONS,
    FINGERTIP_INDICES,
    FINGER_NAMES,
    THUMB_TIP,
    INDEX_TIP,
)

# Color Palette (BGR)
COLOR_BG_DARK = (20, 20, 24)
COLOR_ACCENT_CYAN = (235, 206, 0)      # Neon Cyan / Electric Blue (BGR: 0, 206, 235) -> Let's use (240, 180, 0)
COLOR_CYAN = (255, 220, 0)
COLOR_EMERALD = (80, 220, 100)         # Neon Green (BGR)
COLOR_CORAL = (80, 90, 255)            # Bright Red/Coral
COLOR_GOLD = (50, 190, 255)            # Bright Gold/Orange
COLOR_PURPLE = (255, 100, 180)         # Neon Purple/Magenta
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (140, 140, 140)
COLOR_DARK_PANEL = (25, 25, 30)


def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=15):
    """Draws a tech-style corner bracket rectangle around a bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)

    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)

    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)

    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_hand_skeleton(frame, hand_data, draw_joints=True, draw_fingertips=True):
    """
    Renders high-contrast joints and bones on the hand.
    """
    pts = hand_data["pixel_landmarks"]
    fingers_up = hand_data["fingers_up"]

    # 1. Draw connection bones
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        pt1 = pts[p1_idx]
        pt2 = pts[p2_idx]
        cv2.line(frame, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.line(frame, pt1, pt2, COLOR_CYAN, 1, cv2.LINE_AA)

    # 2. Draw standard joints
    if draw_joints:
        for idx, (px, py) in enumerate(pts):
            if idx not in FINGERTIP_INDICES:
                cv2.circle(frame, (px, py), 4, COLOR_BG_DARK, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 3, COLOR_GOLD, -1, cv2.LINE_AA)

    # 3. Highlight fingertips with dynamic color (Green if extended, Red/Orange if folded)
    if draw_fingertips:
        for f_idx, tip_idx in enumerate(FINGERTIP_INDICES):
            px, py = pts[tip_idx]
            is_up = fingers_up[f_idx]
            tip_color = COLOR_EMERALD if is_up else COLOR_CORAL

            # Outer aura ring
            cv2.circle(frame, (px, py), 8, tip_color, 2, cv2.LINE_AA)
            # Inner solid core
            cv2.circle(frame, (px, py), 5, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_pinch_overlay(frame, hand_data):
    """Visualizes pinch gesture between Thumb and Index finger."""
    pinch = hand_data["pinch"]
    t_px = pinch["thumb_tip_px"]
    i_px = pinch["index_tip_px"]
    c_px = pinch["center_px"]
    is_pinching = pinch["is_pinching"]

    color = COLOR_EMERALD if is_pinching else COLOR_PURPLE
    thickness = 3 if is_pinching else 1

    cv2.line(frame, t_px, i_px, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, c_px, 6 if is_pinching else 4, color, -1, cv2.LINE_AA)

    if is_pinching:
        cv2.putText(
            frame,
            "PINCH!",
            (c_px[0] + 10, c_px[1] - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            COLOR_EMERALD,
            1,
            cv2.LINE_AA,
        )


def apply_clahe_contrast_enhancement(frame):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the luminance channel of the frame to enhance palm details
    under harsh or overexposed lighting.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def draw_hand_badge(frame, hand_data):
    """Draws a floating badge directly over the hand bounding box."""
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = hand_data["bbox"]
    handedness = hand_data["handedness"].upper()
    orient_short = "PALM" if hand_data.get("is_front_palm") else "BACK"
    gesture_name = hand_data["gesture_display"].upper()
    score = int(hand_data["gesture_score"] * 100)

    # Corner box around the hand
    box_color = COLOR_EMERALD if hand_data.get("is_front_palm") else COLOR_CYAN
    draw_corner_rect(frame, (x_min, y_min), (x_max, y_max), box_color, thickness=2, corner_len=18)

    # Floating label pill
    label_text = f"{handedness} [{orient_short}] | {gesture_name} ({score}%)"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.43, 1)

    pill_x1 = max(10, min(x_min, w - tw - 24))
    pill_x2 = pill_x1 + tw + 16

    # Place below bounding box if hand is near the top edge
    if y_min < 55:
        pill_y1 = y_max + 8
    else:
        pill_y1 = max(45, y_min - 25)
    pill_y2 = pill_y1 + th + 10

    # Draw semi-transparent pill backdrop
    sub_img = frame[pill_y1:pill_y2, pill_x1:pill_x2]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        dark_rect = np.full(sub_img.shape, COLOR_DARK_PANEL, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.3, dark_rect, 0.7, 0, sub_img)

    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), box_color, 1)
    cv2.putText(
        frame,
        label_text,
        (pill_x1 + 8, pill_y1 + th + 4),
        cv2.FONT_HERSHEY_DUPLEX,
        0.43,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, detected_hands, fps=0.0, show_help=True):
    """
    Renders top status bar and side info cards for all detected hands.
    """
    h, w, _ = frame.shape

    # 1. Top Status Header Bar
    bar_height = 42
    header_overlay = np.full((bar_height, w, 3), COLOR_DARK_PANEL, dtype=np.uint8)
    frame[0:bar_height, 0:w] = cv2.addWeighted(frame[0:bar_height, 0:w], 0.25, header_overlay, 0.75, 0)
    cv2.line(frame, (0, bar_height), (w, bar_height), COLOR_CYAN, 1)

    # Responsive title sizing
    if w >= 720:
        title_text = "HAND & GESTURE RECOGNITION AI"
        font_scale = 0.60
    elif w >= 500:
        title_text = "HAND & GESTURE AI"
        font_scale = 0.50
    else:
        title_text = "HAND AI"
        font_scale = 0.45

    cv2.putText(
        frame,
        title_text,
        (14, 27),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    # FPS Indicator (Right-aligned)
    fps_color = COLOR_EMERALD if fps >= 25 else (COLOR_GOLD if fps >= 15 else COLOR_CORAL)
    fps_text = f"FPS: {fps:4.1f}"
    (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
    fps_x = max(10, w - fps_w - 14)
    cv2.putText(
        frame,
        fps_text,
        (fps_x, 27),
        cv2.FONT_HERSHEY_DUPLEX,
        0.48,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Hands count tag (Left of FPS)
    hands_text = f"HANDS: {len(detected_hands)}"
    (hands_w, _), _ = cv2.getTextSize(hands_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
    hands_x = fps_x - hands_w - 18
    if hands_x > 180:
        cv2.putText(
            frame,
            hands_text,
            (hands_x, 27),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            COLOR_CYAN,
            1,
            cv2.LINE_AA,
        )

    # 2. Side Panels for each detected hand (Left / Bottom-Left)
    card_width = min(230, w - 30)
    card_height = 135
    margin = 15

    for idx, hand in enumerate(detected_hands):
        card_x = margin + idx * (card_width + 10)
        card_y = h - card_height - (32 if show_help else 15)

        if card_x + card_width > w:
            break

        # Draw translucent background card
        sub = frame[card_y:card_y + card_height, card_x:card_x + card_width]
        if sub.shape[0] == card_height and sub.shape[1] == card_width:
            dark_card = np.full(sub.shape, COLOR_DARK_PANEL, dtype=np.uint8)
            cv2.addWeighted(sub, 0.2, dark_card, 0.8, 0, sub)

        card_border_color = COLOR_EMERALD if hand.get("is_front_palm") else COLOR_CYAN
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_width, card_y + card_height),
            card_border_color,
            1,
        )

        # Header: Hand identity & orientation
        handedness_label = f"{hand['handedness'].upper()} HAND"
        cv2.putText(
            frame,
            handedness_label,
            (card_x + 8, card_y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        orient_label = hand.get("orientation_desc", "FRONT (PALM)")
        orient_color = COLOR_EMERALD if hand.get("is_front_palm") else COLOR_CYAN
        cv2.putText(
            frame,
            orient_label,
            (card_x + card_width - 105, card_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            orient_color,
            1,
            cv2.LINE_AA,
        )

        # Detected Gesture
        gesture_label = hand["gesture_display"].upper()
        conf_pct = int(hand["gesture_score"] * 100)
        cv2.putText(
            frame,
            f"GESTURE: {gesture_label}",
            (card_x + 10, card_y + 46),
            cv2.FONT_HERSHEY_DUPLEX,
            0.43,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # Confidence Bar
        bar_x = card_x + 10
        bar_y = card_y + 54
        bar_w = card_width - 20
        bar_h = 6
        fill_w = int(bar_w * (conf_pct / 100.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COLOR_EMERALD, -1)

        # Finger count
        f_count = hand["finger_count"]
        cv2.putText(
            frame,
            f"FINGERS EXTENDED: {f_count} / 5",
            (card_x + 10, card_y + 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            COLOR_GOLD,
            1,
            cv2.LINE_AA,
        )

        # 5-Finger Indicators (T, I, M, R, P)
        fingers_up = hand["fingers_up"]
        labels = ["T", "I", "M", "R", "P"]
        badge_start_x = card_x + 10
        badge_y = card_y + 115
        slot_w = (card_width - 20) // 5

        for fi, (char, is_up) in enumerate(zip(labels, fingers_up)):
            bx = badge_start_x + fi * slot_w
            badge_box_w = slot_w - 6
            f_color = COLOR_EMERALD if is_up else COLOR_CORAL
            bg_color = (35, 60, 35) if is_up else (35, 35, 60)

            # Mini badge
            cv2.rectangle(frame, (bx, badge_y - 16), (bx + badge_box_w, badge_y + 6), bg_color, -1)
            cv2.rectangle(frame, (bx, badge_y - 16), (bx + badge_box_w, badge_y + 6), f_color, 1)

            cv2.putText(
                frame,
                f"{char}:{'1' if is_up else '0'}",
                (bx + 4, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                f_color,
                1,
                cv2.LINE_AA,
            )

    # 3. Bottom Keyboard Shortcuts Ribbon
    if show_help:
        bottom_h = 24
        b_overlay = np.full((bottom_h, w, 3), (15, 15, 18), dtype=np.uint8)
        frame[h - bottom_h:h, 0:w] = cv2.addWeighted(
            frame[h - bottom_h:h, 0:w], 0.2, b_overlay, 0.8, 0
        )
        shortcuts_text = "[S] SKELETON  [B] BOX  [C] CONTRAST BOOST  [P] PINCH  [H] HUD  [M] MIRROR  [SPACE] PAUSE  [ENTER] SNAP  [Q] QUIT"
        cv2.putText(
            frame,
            shortcuts_text,
            (12, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_GRAY,
            1,
            cv2.LINE_AA,
        )
