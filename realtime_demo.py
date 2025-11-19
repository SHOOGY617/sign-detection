# realtime_demo.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import threading

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# load model
m = joblib.load("model.joblib")
clf = m["model"]
le = m["le"]

engine = pyttsx3.init()
engine.setProperty('rate', 160)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    last_spoken = ""
    last_time = 0
    last_label = None

    def speak_async(text):
        def _s():
            engine.say(text)
            engine.runAndWait()
        t = threading.Thread(target=_s, daemon=True)
        t.start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        display_text = "No hand"
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            coords = []
            for p in lm.landmark:
                coords.append(p.x)
                coords.append(p.y)
            # Convert to same feature format as training: x0..x20, y0..y20
            xs = coords[0::2]
            ys = coords[1::2]
            feat = np.array(xs + ys).reshape(1, -1)
            pred = clf.predict(feat)[0]
            label = le.inverse_transform([pred])[0]
            display_text = label

            mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            # Speak whenever the detected label changes (reset when no hand)
            if last_label is None or label != last_label:
                # use non-blocking speech so video doesn't freeze
                speak_async(label)
                last_spoken = label
                last_time = time.time()
                last_label = label

        cv2.putText(img, f"Detected: {display_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Sign2Speech", img)
        key = cv2.waitKey(1) & 0xFF
        # ESC or 'q' to quit
        if key == 27 or key == ord('q'):
            break

        # if no hand detected, reset last_label so next detection is considered a change
        if not results.multi_hand_landmarks:
            last_label = None

cap.release()
cv2.destroyAllWindows()
