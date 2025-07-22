
import cv2
import mediapipe as mp
import time
from datetime import datetime
import math
from plyer import notification
import threading

# Desktop notification
def show_alert(message):
    threading.Thread(
        target=lambda: notification.notify(
            title="Gesture Alert",
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
    mp_face_mesh = mp.solutions.face_mesh

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    SHAKE_THRESHOLD = 0.06
    TOO_LONG_VISIBLE_SEC = 4
    NOTIFY_COOLDOWN = 3

    prev_hand_pos = {}
    hand_visible_start = {}
    last_alert_time = 0
    last_print_time = {}

    # Eye tracking constants
    LEFT_IRIS = [474, 475, 476, 477]
    LEFT_EYE = [33, 133]
    EYE_LIDS = [159, 145, 386, 374]
    HORIZONTAL_RANGE = (0.25, 0.75)
    VERTICAL_RANGE = (0.3, 0.7)
    EYE_AWAY_DURATION = 2.0
    EYE_ALERT_COOLDOWN = 5.0

    eye_away_start = None
    last_eye_alert_time = 0

    print("🖐️👁️ Gesture (Hand + Eye) tracking active.")

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
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

                if not is_above_head and prev_pos:
                    dist = calc_distance(prev_pos, wrist)
                    if dist > SHAKE_THRESHOLD:
                        if current_time - last_print_time.get(hand_id, 0) >= 2:
                            print(f"⚠️ Excessive hand movement at {datetime.now().strftime('%H:%M:%S')}")
                            last_print_time[hand_id] = current_time
                        alert_triggered = True

                if hand_id not in hand_visible_start:
                    hand_visible_start[hand_id] = current_time
                else:
                    visible_duration = current_time - hand_visible_start[hand_id]
                    if visible_duration >= TOO_LONG_VISIBLE_SEC:
                        if current_time - last_print_time.get(f"{hand_id}_long", 0) >= 2:
                            print(f"⚠️ Hand visible too long at {datetime.now().strftime('%H:%M:%S')}")
                            last_print_time[f"{hand_id}_long"] = current_time
                        alert_triggered = True

        for hand_id in list(hand_visible_start.keys()):
            if hand_id not in visible_hands:
                hand_visible_start.pop(hand_id, None)
                prev_hand_pos.pop(hand_id, None)
                last_print_time.pop(hand_id, None)
                last_print_time.pop(f"{hand_id}_long", None)

        if alert_triggered and visible_hands and (current_time - last_alert_time >= NOTIFY_COOLDOWN):
            show_alert("⚠️ Hand movement detected!")
            last_alert_time = current_time

        # Eye Contact Detection
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark

            eye_left = face_landmarks[LEFT_EYE[0]].x
            eye_right = face_landmarks[LEFT_EYE[1]].x
            iris_center_x = sum([face_landmarks[i].x for i in LEFT_IRIS]) / len(LEFT_IRIS)
            iris_rel_x = (iris_center_x - eye_left) / (eye_right - eye_left)

            upper_lid = (face_landmarks[EYE_LIDS[0]].y + face_landmarks[EYE_LIDS[2]].y) / 2
            lower_lid = (face_landmarks[EYE_LIDS[1]].y + face_landmarks[EYE_LIDS[3]].y) / 2
            iris_center_y = sum([face_landmarks[i].y for i in LEFT_IRIS]) / len(LEFT_IRIS)
            iris_rel_y = (iris_center_y - upper_lid) / (lower_lid - upper_lid)

            # Manually detected bounds for valid eye contact values
            is_away = not (2.65 <= iris_rel_x <= 2.95 and 0.03 <= iris_rel_y <= 0.26)


            if is_away:
                if eye_away_start is None:
                    eye_away_start = current_time
                elif (current_time - eye_away_start >= EYE_AWAY_DURATION):
                    if current_time - last_eye_alert_time >= EYE_ALERT_COOLDOWN:
                        print(f"⚠️ Eye contact lost at {datetime.now().strftime('%H:%M:%S')}")
                        show_alert("⚠️ Eye contact lost!")
                        last_eye_alert_time = current_time
            else:
                eye_away_start = None