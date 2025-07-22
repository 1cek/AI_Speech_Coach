import cv2

def start_stream(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("📷 Camera stream started.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()
    print("📷 Camera stream stopped.")
