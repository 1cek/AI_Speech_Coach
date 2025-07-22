def run():
    import sounddevice as sd
    import queue
    import json
    from vosk import Model, KaldiRecognizer
    from datetime import datetime
    from plyer import notification

    # Desktop notification
    def show_alert(message):
        notification.notify(
            title="Filler Word Alert",
            message=message,
            app_name='Speech Coach',
            timeout=5  # seconds
        )

    # Load Vosk model
    model_path = r"D:\AI project\Presenter_Coach\vosk-model-small-en-us-0.15"
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    # Define filler words to detect
    filler_words = {
        "uh", "um", "äh", "ähm", "you know", "actually",
        "basically", "ehh", "I mean", "well", "yeah", "eeh", "aahh", "oooh"
    }

    # Audio queue
    q = queue.Queue()
    last_detected = set()

    # Callback for sounddevice input
    def callback(indata, frames, time, status):
        if status:
            print("⚠️ Audio Status:", status, flush=True)
        q.put(bytes(indata))

    print("🎙️ Speak now! (Ctrl+C to stop)")

    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if "result" in result:
                        current_detected = set()
                        for word_obj in result["result"]:
                            word = word_obj["word"].lower()
                            if word in filler_words:
                                if word not in current_detected:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    print(f"⚠️ FILLER WORD DETECTED at {timestamp}: {word.upper()}", flush=True)
                                    show_alert(f"Filler word detected: {word.upper()}")
                                    current_detected.add(word)
                        last_detected = set()
                else:
                    partial_result = json.loads(recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "").lower()
                    for word in filler_words:
                        if word in partial_text and word not in last_detected:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"⚠️ FILLER WORD DETECTED at {timestamp}: {word.upper()}", flush=True)
                            show_alert(f"⚠️ Filler word detected: {word.upper()}")
                            last_detected.add(word)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")