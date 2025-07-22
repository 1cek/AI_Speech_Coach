
import cv2
import mediapipe as mp
import time
from datetime import datetime
import math
from plyer import notification
import threading

# Desktop notification helper
def show_alert(message):
    threading.Thread(
        target=lambda: notification.notify(
            title="Hand Movement Alert",
            message=message,
            app_name='Speech Coach',
            timeout=5
        ),
        daemon=True
    ).start()

def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def run(frame_queue, stop_event):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    SHAKE_THRESHOLD = 0.06
    TOO_LONG_VISIBLE_SEC = 3
    NOTIFY_COOLDOWN = 3

    prev_hand_pos = {}
    hand_visible_start = {}
    last_alert_time = 0
    last_print_time = {}

    print("🖐️ Hand gesture tracking active.")

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        current_time = time.time()

        head_y = None
        if pose_results.pose_landmarks:
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            head_y = nose.y

        visible_hands = set()
        alert_triggered = False

        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_id = f"hand_{idx}"
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                visible_hands.add(hand_id)
                hand_y = wrist.y
                is_above_head = head_y is not None and hand_y < head_y
                prev_pos = prev_hand_pos.get(hand_id)
                prev_hand_pos[hand_id] = wrist

                # 1. Excessive shaking
                if not is_above_head and prev_pos:
                    dist = calc_distance(prev_pos, wrist)
                    if dist > SHAKE_THRESHOLD:
                        if current_time - last_print_time.get(hand_id, 0) >= 2:
                            print(f"⚠️ Excessive hand movement at {datetime.now().strftime('%H:%M:%S')}")
                            last_print_time[hand_id] = current_time
                        alert_triggered = True

                # 2. Visible too long
                if hand_id not in hand_visible_start:
                    hand_visible_start[hand_id] = current_time
                else:
                    visible_duration = current_time - hand_visible_start[hand_id]
                    if visible_duration >= TOO_LONG_VISIBLE_SEC:
                        if current_time - last_print_time.get(f"{hand_id}_long", 0) >= 2:
                            print(f"⚠️ Hand visible too long at {datetime.now().strftime('%H:%M:%S')}")
                            last_print_time[f"{hand_id}_long"] = current_time
                        alert_triggered = True

        # Clean up disappeared hands
        for hand_id in list(hand_visible_start.keys()):
            if hand_id not in visible_hands:
                hand_visible_start.pop(hand_id, None)
                prev_hand_pos.pop(hand_id, None)
                last_print_time.pop(hand_id, None)
                last_print_time.pop(f"{hand_id}_long", None)

        # Desktop notification logic (global cooldown & only if hand still visible)
        if alert_triggered and visible_hands and (current_time - last_alert_time >= NOTIFY_COOLDOWN):
            show_alert("⚠️ Hand movement detected!")
            last_alert_time = current_time
