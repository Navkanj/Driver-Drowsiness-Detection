import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array  # safer import path across TF versions
from threading import Thread,Lock
import time

# -------------------- Config (tweak here) --------------------
MODEL_PATH = "drowiness_new7.h5"
ALARM_PATH = "assets/alarm.mp3"

audio_guard = Lock()
audio_playing_until = 0.0

EAR_THRESHOLD = 0.22          # eye aspect ratio threshold
MAR_THRESHOLD = 0.90          # yawn threshold
CONSEC_FRAMES = 10            # frames of low EAR to trigger eye drowsiness
YAWN_ALERT_THRESHOLD = 3      # yawns before yawn-based alert
ALARM_COOLDOWN = 2.0          # seconds between sound plays
SOUND_DURATION = 3.0          # seconds sound is assumed to play (for spacing)
YAWN_RESET_TIME = 10.0        # reset yawn counter if no yawns for this many seconds
RECOVERY_FRAMES = 15          # frames of normal EAR to clear eye-drowsy state

# -------------------- Load heavy stuff ONCE --------------------
# Mediapipe FaceMesh (add confidences; set refine_landmarks=False if unstable on your machine)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,          # flip to False if you see FaceMesh init errors
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Your trained eye-state CNN
model = load_model(MODEL_PATH)
CLASSES = ['yawn', 'no_yawn', 'Closed', 'Open']  # kept for display

# Landmark indices
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]
MOUTH_INDICES     = [61, 291, 81, 178, 13, 14, 312, 308]

# -------------------- Helpers --------------------
def safe_play_alarm(path: str):
    # ensure only one playback within a 2.5s window
    global audio_playing_until
    try:
        now = time.time()
        with audio_guard:
            if now < audio_playing_until:
                return  # a recent/ongoing playback; skip
            audio_playing_until = now + 4  # lock window: 2.5s

        from playsound import playsound
        playsound(path)  # blocking inside thread; that's fine
    except Exception:
        pass


def compute_EAR(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))

def compute_MAR(landmarks, mouth_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in mouth_indices]
    return (np.linalg.norm(p[2]-p[6]) + np.linalg.norm(p[3]-p[7]) + np.linalg.norm(p[4]-p[5])) / (2.0 * np.linalg.norm(p[0]-p[1]))

def rect_ok(x, y, w, h, W, H):
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= W and y + h <= H

# -------------------- Main generator --------------------
def run_drowsiness_detection(model_path: str = MODEL_PATH, alarm_path: str = ALARM_PATH):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    # State
    blink_frames = 0
    recovery_frames = 0
    yawn_count = 0
    yawn_active = False
    last_yawn_time = 0.0
    last_sound_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            H, W = frame.shape[:2]
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            EAR_avg = 0.0
            MAR = 0.0
            left_pred = "N/A"
            right_pred = "N/A"

            is_eye_drowsy = False
            is_yawn_drowsy = False
            alarm_on = False  # what we return to the UI as "currently drowsy"
            now = time.time()

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # EAR & MAR
                EAR_left  = compute_EAR(lm, LEFT_EYE_INDICES)
                EAR_right = compute_EAR(lm, RIGHT_EYE_INDICES)
                EAR_avg   = (EAR_left + EAR_right) / 2.0
                MAR       = compute_MAR(lm, MOUTH_INDICES)

                # CNN eye crops (defensive bounds)
                try:
                    def crop_and_predict(idx):
                        coords = np.array([[lm[i].x * W, lm[i].y * H] for i in idx]).astype(int)
                        x, y, w, h = cv2.boundingRect(coords)
                        if not rect_ok(x, y, w, h, W, H):
                            return "CropFail"
                        img = frame[y:y+h, x:x+w]
                        img = cv2.resize(img, (145, 145))
                        img = np.expand_dims(img_to_array(img.astype('float32') / 255.0), axis=0)
                        pred = np.argmax(model.predict(img))
                        return CLASSES[pred]
                    left_pred  = crop_and_predict(LEFT_EYE_INDICES)
                    right_pred = crop_and_predict(RIGHT_EYE_INDICES)
                except Exception:
                    left_pred = right_pred = "Error"

                # --- Eye drowsiness (EAR) with debounce ---
                if EAR_avg < EAR_THRESHOLD:
                    blink_frames += 1
                    recovery_frames = 0
                else:
                    recovery_frames += 1
                    if recovery_frames >= RECOVERY_FRAMES:
                        blink_frames = 0

                if blink_frames >= CONSEC_FRAMES:
                    is_eye_drowsy = True

                # --- Yawn drowsiness (MAR) with edge-trigger and decay ---
                if MAR > MAR_THRESHOLD and not yawn_active:
                    yawn_count += 1
                    yawn_active = True
                    last_yawn_time = now
                elif MAR <= MAR_THRESHOLD:
                    yawn_active = False

                if now - last_yawn_time > YAWN_RESET_TIME:
                    yawn_count = 0

                if yawn_count >= YAWN_ALERT_THRESHOLD:
                    is_yawn_drowsy = True

                # --- Combine status for UI ---
                alarm_on = is_eye_drowsy or is_yawn_drowsy

                # --- Trigger alarm; guard inside safe_play_alarm prevents overlap ---
                want_sound = is_eye_drowsy or is_yawn_drowsy
                if want_sound:
                    Thread(target=safe_play_alarm, args=(alarm_path,), daemon=True).start()



            # (Optional) overlay if you run this file directly
            # cv2.putText(frame, f"EAR:{EAR_avg:.2f} MAR:{MAR:.2f} Yawns:{yawn_count}", (8, 24),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            yield frame, EAR_avg, MAR, yawn_count, alarm_on, left_pred, right_pred

    finally:
        cap.release()
