import streamlit as st
import cv2
import numpy as np
import pandas as pd
from detect_drowsiness import run_drowsiness_detection

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("**Hybrid CNN + Mediapipe-based Real-Time Detection**")

FRAME_WINDOW = st.image([])
status_placeholder = st.empty()
stats_placeholder = st.empty()

trip_data = []
trip_running = st.session_state.get("trip_running", False)

col1, col2 = st.columns(2)
with col1:
    start = st.button("▶️ Start Trip", use_container_width=True)
with col2:
    stop = st.button("⏹️ Stop Trip", use_container_width=True)

if start:
    st.session_state.trip_running = True
    st.success("Trip Started. Monitoring driver alertness...")

if stop:
    st.session_state.trip_running = False
    st.warning("Trip Stopped. Generating report...")

if st.session_state.get("trip_running", False):
    for frame, EAR_avg, MAR, yawn_count, alarm_on, left_pred, right_pred in run_drowsiness_detection():
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        status_text = "⚠️ DROWSY!" if alarm_on else "✅ Awake"
        status_color = "red" if alarm_on else "green"
        status_placeholder.markdown(f"### **Status:** <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)

        stats_placeholder.write({
            "EAR": round(EAR_avg, 3),
            "MAR": round(MAR, 3),
            "Yawn Count": yawn_count,
            "Left Eye": left_pred,
            "Right Eye": right_pred
        })

        trip_data.append([EAR_avg, MAR, yawn_count, left_pred, right_pred, "Drowsy" if alarm_on else "Awake"])
else:
    if trip_data:
        df = pd.DataFrame(trip_data, columns=["EAR", "MAR", "Yawn Count", "Left Eye", "Right Eye", "Status"])
        df.to_csv("trip_report.csv", index=False)
        st.dataframe(df.tail(10))
        st.success("Trip report saved as **trip_report.csv**.")
