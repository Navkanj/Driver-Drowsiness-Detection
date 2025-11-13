import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array  # safer import path across TF versions
from threading import Thread,Lock
import time
import serial

# -------------------- Arduino Serial --------------------
try:
    arduino = serial.Serial('COM11', 9600)   # ⚠️ Change COM port if needed
    time.sleep(2)
    print("✅ Arduino Connected")
except Exception as e:
    arduino = None
    print("⚠️ Arduino Not Connected:", e)

# -------------------- Config (tweak here) --------------------
MODEL_PATH = "drowiness_new7.h5"
ALARM_PATH = "assets/alarm.mp3"

audio_guard = Lock()
audio_playing_until = 0.0

EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.90
CONSEC_FRAMES = 10
YAWN_ALERT_THRESHOLD = 3
ALARM_COOLDOWN = 2.0
SOUND_DURATION = 3.0
YAWN_RESET_TIME = 10.0
RECOVERY_FRAMES = 15

# -------------------- Load once --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = load_model(MODEL_PATH)
CLASSES = ['yawn', 'no_yawn', 'Closed', 'Open']

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]
MOUTH_INDICES     = [61, 291, 81, 178, 13, 14, 312, 308]

# -------------------- Helpers --------------------
def safe_play_alarm(path: str):
    global audio_playing_until
    try:
        now = time.time()
        with audio_guard:
            if now < audio_playing_until:
                return
            audio_playing_until = now + 4  

        from playsound import playsound
        playsound(path)
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

    blink_frames = 0
    recovery_frames = 0
    yawn_count = 0
    yawn_active = False
    last_yawn_time = 0.0

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
            alarm_on = False
            now = time.time()

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                EAR_left  = compute_EAR(lm, LEFT_EYE_INDICES)
                EAR_right = compute_EAR(lm, RIGHT_EYE_INDICES)
                EAR_avg   = (EAR_left + EAR_right) / 2.0
                MAR       = compute_MAR(lm, MOUTH_INDICES)

                try:
                    def crop_and_predict(idx):
                        coords = np.array([[lm[i].x*W, lm[i].y*H] for i in idx]).astype(int)
                        x,y,w,h = cv2.boundingRect(coords)
                        if not rect_ok(x,y,w,h,W,H):
                            return "CropFail"
                        img = frame[y:y+h, x:x+w]
                        img = cv2.resize(img, (145,145))
                        img = np.expand_dims(img_to_array(img.astype('float32')/255.0), axis=0)
                        pred = np.argmax(model.predict(img))
                        return CLASSES[pred]
                    left_pred  = crop_and_predict(LEFT_EYE_INDICES)
                    right_pred = crop_and_predict(RIGHT_EYE_INDICES)
                except Exception:
                    left_pred = right_pred = "Error"

                if EAR_avg < EAR_THRESHOLD:
                    blink_frames += 1
                    recovery_frames = 0
                else:
                    recovery_frames += 1
                    if recovery_frames >= RECOVERY_FRAMES:
                        blink_frames = 0

                if blink_frames >= CONSEC_FRAMES:
                    is_eye_drowsy = True

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

                alarm_on = is_eye_drowsy or is_yawn_drowsy

                if alarm_on:
                    Thread(target=safe_play_alarm, args=(alarm_path,), daemon=True).start()

                # -------------------- Send to Arduino --------------------
                if arduino:
                    if alarm_on:
                        arduino.write(b"DROWSY\n")
                    else:
                        arduino.write(b"SAFE\n")

            yield frame, EAR_avg, MAR, yawn_count, alarm_on, left_pred, right_pred

    finally:
        cap.release()
