
import multiprocessing
import filler_word_detector  # Vosk-based script
import hand_eye_tracker      # Hand and eye tracker using frame_queue
import camera_streamer       # Provides webcam frames to hand_eye_tracker

def run_filler_detector():
    filler_word_detector.run() 

def run_camera(frame_queue, stop_event):
    camera_streamer.start_stream(frame_queue, stop_event)

def run_hand_eye_tracker(frame_queue, stop_event):
    hand_eye_tracker.run(frame_queue, stop_event)

def main():
    # Shared webcam stream between camera and hand tracker
    frame_queue = multiprocessing.Queue(maxsize=5)
    stop_event = multiprocessing.Event()

    # Define processes
    p_mic = multiprocessing.Process(target=run_filler_detector)
    p_cam = multiprocessing.Process(target=run_camera, args=(frame_queue, stop_event))
    p_hand = multiprocessing.Process(target=run_hand_eye_tracker, args=(frame_queue, stop_event))

    processes = [p_mic, p_cam, p_hand]

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected! Stopping modules...")
        stop_event.set()
        for p in processes:
            p.terminate()
            p.join()
        print("✅ All modules stopped cleanly.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Needed on Windows
    main()
