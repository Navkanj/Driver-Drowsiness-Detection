import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread
import time
import threading

def start_alarm(sound):
    try:
        playsound(sound)
    except Exception as e:
        print(f"Error playing sound: {str(e)}")

def compute_EAR(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))

def compute_MAR(landmarks, mouth_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in mouth_indices]
    return (np.linalg.norm(p[2]-p[6]) + np.linalg.norm(p[3]-p[7]) + np.linalg.norm(p[4]-p[5])) / (2.0 * np.linalg.norm(p[0]-p[1]))

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

    # Define landmarks indices at the top level
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]
    MOUTH_INDICES = [61, 291, 81, 178, 13, 14, 312, 308]

    # Modified alarm variables
    SOUND_DURATION = 3  # reduced duration for quicker replay
    last_sound_end_time = 0
    alarm_playing = False
    
    # Modified state variables
    is_eye_drowsy = False
    is_yawn_drowsy = False
    recovery_frames = 0
    RECOVERY_THRESHOLD = 15  # increased threshold for more stability
    YAWN_RESET_TIME = 10  # Reset yawn counter after 30 seconds of no yawns
    last_yawn_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        EAR_avg = MAR = 0
        left_pred = right_pred = "N/A"
        current_time = time.time()  # Move time.time() outside conditional

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Calculate EAR and MAR using the unified functions
                EAR_left = compute_EAR(landmarks, LEFT_EYE_INDICES)
                EAR_right = compute_EAR(landmarks, RIGHT_EYE_INDICES)
                EAR_avg = (EAR_left + EAR_right) / 2.0
                MAR = compute_MAR(landmarks, MOUTH_INDICES)

                try:
                    # Eye region extraction and prediction
                    left_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_INDICES])
                    right_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_INDICES])
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
                except Exception as e:
                    print(f"Error processing eyes: {str(e)}")
                    left_pred = right_pred = "Error"

                current_time = time.time()

                # Modified eye drowsiness detection with immediate alarm
                if EAR_avg < EAR_THRESHOLD:
                    count += 1
                    recovery_frames = 0
                    # Check if we should play alarm again
                    if count >= CONSEC_FRAMES and current_time >= last_sound_end_time:
                        is_eye_drowsy = True
                        Thread(target=start_alarm, args=(alarm_path,), daemon=True).start()
                        last_sound_end_time = current_time + SOUND_DURATION
                else:
                    recovery_frames += 1
                    if recovery_frames >= RECOVERY_THRESHOLD:
                        count = 0
                        is_eye_drowsy = False

                # Modified yawn detection with timeout
                if MAR > MAR_THRESHOLD and not yawn_active:
                    yawn_count += 1
                    yawn_active = True
                    last_yawn_time = current_time
                    is_yawn_drowsy = yawn_count > YAWN_ALERT_THRESHOLD
                elif MAR <= MAR_THRESHOLD:
                    yawn_active = False

                # Reset yawn count if no yawns for YAWN_RESET_TIME
                if current_time - last_yawn_time > YAWN_RESET_TIME:
                    yawn_count = 0
                    is_yawn_drowsy = False

                # Modified alarm logic - separate for eyes and yawns
                should_alert_eyes = count >= CONSEC_FRAMES
                should_alert_yawns = yawn_count > YAWN_ALERT_THRESHOLD

                # Yawn alarm logic
                if should_alert_yawns and not alarm_on and (current_time - last_alarm_time > ALARM_COOLDOWN):
                    alarm_on = True
                    last_alarm_time = current_time
                    Thread(target=start_alarm, args=(alarm_path,), daemon=True).start()
                    is_yawn_drowsy = True

                # Reset alarm state only for yawns
                if not should_alert_yawns:
                    is_yawn_drowsy = False
                    alarm_on = False

        # Modified display code
        if results.multi_face_landmarks:
            cv2.putText(frame, f"EAR: {EAR_avg:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Yawns: {yawn_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"L: {left_pred} | R: {right_pred}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            status = "DROWSY (Eyes)" if is_eye_drowsy else "DROWSY (Yawning)" if is_yawn_drowsy else "AWAKE"
            color = (0, 0, 255) if (is_eye_drowsy or is_yawn_drowsy) else (0, 255, 0)
            cv2.putText(frame, f"Status: {status}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        yield frame, EAR_avg, MAR, yawn_count, alarm_on, left_pred, right_pred

    cap.release()
