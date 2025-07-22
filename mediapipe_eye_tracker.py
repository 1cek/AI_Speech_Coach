import cv2
import mediapipe as mp
import time
from datetime import datetime
from plyer import notification

# Constants
AWAY_DURATION = 2.0
EYE_CLOSED_GAP = 0.01
HORIZONTAL_RANGE = (0.25, 0.75)
VERTICAL_RANGE = (0.3, 0.7)
NOTIFY_COOLDOWN = 5.0

# MediaPipe config
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 133]
EYE_LIDS = [159, 145, 386, 374]

def is_eyes_closed(landmarks):
    upper_avg = (landmarks[0].y + landmarks[2].y) / 2
    lower_avg = (landmarks[1].y + landmarks[3].y) / 2
    return abs(upper_avg - lower_avg) < EYE_CLOSED_GAP

def show_alert(message):
    notification.notify(
        title="Eye Contact Alert",
        message=message,
        app_name="Speech Coach",
        timeout=4
    )

def process_frame(frame, face_mesh, state):
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    current_time = time.time()

    is_away = False

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        if not state["face_stable"]:
            if state["stable_start_time"] is None:
                state["stable_start_time"] = current_time
            elif current_time - state["stable_start_time"] >= 1.5:
                state["face_stable"] = True
        elif state["face_stable"]:
            eye_left = face_landmarks[LEFT_EYE[0]].x
            eye_right = face_landmarks[LEFT_EYE[1]].x
            iris_center_x = sum([face_landmarks[i].x for i in LEFT_IRIS]) / len(LEFT_IRIS)
            iris_rel_x = (iris_center_x - eye_left) / (eye_right - eye_left)

            upper_lid = (face_landmarks[EYE_LIDS[0]].y + face_landmarks[EYE_LIDS[2]].y) / 2
            lower_lid = (face_landmarks[EYE_LIDS[1]].y + face_landmarks[EYE_LIDS[3]].y) / 2
            iris_center_y = sum([face_landmarks[i].y for i in LEFT_IRIS]) / len(LEFT_IRIS)
            iris_rel_y = (iris_center_y - upper_lid) / (lower_lid - upper_lid)

            closed = is_eyes_closed([face_landmarks[i] for i in EYE_LIDS])
            is_away = (
                iris_rel_x < HORIZONTAL_RANGE[0] or
                iris_rel_x > HORIZONTAL_RANGE[1] or
                iris_rel_y < VERTICAL_RANGE[0] or
                iris_rel_y > VERTICAL_RANGE[1] or
                closed
            )
    else:
        state["stable_start_time"] = None
        state["face_stable"] = False

    # Alert handling
    if state["face_stable"] and is_away:
        if state["away_start_time"] is None:
            state["away_start_time"] = current_time
        elif not state["alert_shown"] and (current_time - state["away_start_time"]) > AWAY_DURATION:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"⚠️ [ALERT] Eye contact lost at {timestamp}")
            if current_time - state["last_notification_time"] > NOTIFY_COOLDOWN:
                show_alert("⚠️ Eye contact lost!")
                state["last_notification_time"] = current_time
            state["alert_shown"] = True
    else:
        state["away_start_time"] = None
        state["alert_shown"] = False

    return frame

def run(frame_queue=None, stop_event=None):
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    state = {
        "away_start_time": None,
        "alert_shown": False,
        "last_notification_time": 0,
        "face_stable": False,
        "stable_start_time": None
    }

    if frame_queue is None:
        cap = cv2.VideoCapture(0)
        print("👁️ Eye Tracker started standalone — press ESC to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("❌ Failed to grab frame.")
                break

            processed = process_frame(frame, face_mesh, state)
            cv2.imshow("Eye Tracker", processed)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("👁️ Eye Tracker using shared camera")
        while not stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                process_frame(frame, face_mesh, state)
