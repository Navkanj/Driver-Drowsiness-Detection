import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread
import time

def start_alarm(sound):
    playsound(sound)

def compute_EAR(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

def compute_MAR(landmarks, mouth_indices):
    p1, p2, p3, p4, p5, p6, p7, p8 = [np.array([landmarks[i].x, landmarks[i].y]) for i in mouth_indices]
    mar = (np.linalg.norm(p3 - p7) + np.linalg.norm(p4 - p8) + np.linalg.norm(p5 - p6)) / (2.0 * np.linalg.norm(p1 - p2))
    return mar

def run_drowsiness_detection(model_path="drowiness_new7.h5", alarm_path="assets/alarm.mp3"):
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    model = load_model(model_path)
    classes = ['yawn', 'no_yawn', 'Closed', 'Open']

    EAR_THRESHOLD = 0.22
    MAR_THRESHOLD = 0.9
    CONSEC_FRAMES = 10
    YAWN_ALERT_THRESHOLD = 3
    ALARM_COOLDOWN = 2  # seconds before re-triggering alarm

    count = 0
    yawn_count = 0
    alarm_on = False
    yawn_active = False
    last_alarm_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        EAR_avg = MAR = 0
        left_pred = right_pred = "N/A"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [263, 387, 385, 362, 380, 373]
                mouth_idx = [61, 291, 81, 178, 13, 14, 312, 308]

                def compute_EAR(eye_idx):
                    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_idx]
                    return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))

                def compute_MAR(m_idx):
                    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in m_idx]
                    return (np.linalg.norm(p[2]-p[6]) + np.linalg.norm(p[3]-p[7]) + np.linalg.norm(p[4]-p[5])) / (2.0 * np.linalg.norm(p[0]-p[1]))

                EAR_left = compute_EAR(left_eye_idx)
                EAR_right = compute_EAR(right_eye_idx)
                EAR_avg = (EAR_left + EAR_right) / 2.0
                MAR = compute_MAR(mouth_idx)

                try:
                    left_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_eye_idx])
                    right_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_eye_idx])
                    lx, ly, lw, lh = cv2.boundingRect(left_eye_coords.astype(int))
                    rx, ry, rw, rh = cv2.boundingRect(right_eye_coords.astype(int))
                    left_eye_img = cv2.resize(frame[ly:ly+lh, lx:lx+lw], (145, 145))
                    right_eye_img = cv2.resize(frame[ry:ry+rh, rx:rx+rw], (145, 145))
                    left_eye_img = np.expand_dims(img_to_array(left_eye_img.astype('float') / 255.0), axis=0)
                    right_eye_img = np.expand_dims(img_to_array(right_eye_img.astype('float') / 255.0), axis=0)

                    pred1 = np.argmax(model.predict(left_eye_img))
                    pred2 = np.argmax(model.predict(right_eye_img))
                    left_pred = classes[pred1]
                    right_pred = classes[pred2]
                except:
                    left_pred = right_pred = "Error"

                if MAR > MAR_THRESHOLD and not yawn_active:
                    yawn_count += 1
                    yawn_active = True
                elif MAR <= MAR_THRESHOLD:
                    yawn_active = False

                if EAR_avg < EAR_THRESHOLD:
                    count += 1
                else:
                    count = 0

                current_time = time.time()
                should_alert = (count >= CONSEC_FRAMES or yawn_count > YAWN_ALERT_THRESHOLD)

            # Trigger alarm only if cooldown passed and not already active
            if should_alert and not alarm_on and (current_time - last_alarm_time > ALARM_COOLDOWN):
                alarm_on = True
                last_alarm_time = current_time
                Thread(target=start_alarm, args=(alarm_path,), daemon=True).start()

            # Reset alarm after user recovers (normal EAR & MAR)
            elif not should_alert and alarm_on:
                alarm_on = False
                yawn_count = 0

            cv2.putText(frame, f"EAR: {EAR_avg:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Yawns: {yawn_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"L: {left_pred} | R: {right_pred}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        yield frame, EAR_avg, MAR, yawn_count, alarm_on, left_pred, right_pred

    cap.release()
