import streamlit as st
import cv2
import numpy as np
import pandas as pd
from detect_drowsiness import run_drowsiness_detection
import os
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

os.makedirs("reports", exist_ok=True)

### ---------------- GEMINI: DRIVER INSIGHT ---------------- ###
def get_gemini_summary(report):
    if not GOOGLE_API_KEY:
        return "Stay alert and take regular breaks."

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)

        prompt = f"""
You are a professional driver safety coach.

Analyze the driver's drowsiness report and give:
- A clear 2–4 sentence evaluation
- The likely cause (fatigue, distraction etc.)
- 2 specific safety tips
- DO NOT repeat numeric stats, interpret them instead
- Tone: calm and encouraging

Trip data:
{json.dumps(report, indent=2)}
"""
        model = genai.GenerativeModel("gemini-2.0-flash")
        res = model.generate_content(prompt)
        return res.text.strip()
    except:
        return "Rest well and stay alert while driving."

### ---------------------- REPORT BUILD ---------------------- ###
def generate_report(df):
    total_time = len(df) / 30
    drowsy_periods = df[df['Status'] == 'Drowsy']

    return {
        "trip_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_minutes": float(round(total_time / 60, 2)),
        "drowsy_percentage": float(round(len(drowsy_periods) / len(df) * 100, 2)),
        "total_yawns": int(df['Yawn Count'].max()),
        "average_ear": float(round(df['EAR'].mean(), 3)),
        "average_mar": float(round(df['MAR'].mean(), 3)),
        "drowsy_episodes": int((df['Status'] == 'Drowsy').sum())
    }

### ------------------ STREAMLIT UI START ------------------- ###
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")

if 'trip_data' not in st.session_state:
    st.session_state.trip_data = []

FRAME_WINDOW = st.image([])
status_placeholder = st.empty()
stats_placeholder = st.empty()

c1, c2, c3 = st.columns(3)
with c1: start = st.button("▶️ Start Trip", use_container_width=True)
with c2: stop  = st.button("⏹️ Stop Trip", use_container_width=True)
with c3: clear = st.button("🗑️ Clear Data", use_container_width=True)

if start:
    st.session_state.trip_running = True
    st.session_state.trip_data = []
    st.success("Trip Started...")

if stop and len(st.session_state.trip_data) > 0:
    st.session_state.trip_running = False

    df = pd.DataFrame(
        st.session_state.trip_data,
        columns=["EAR", "MAR", "Yawn Count", "Left Eye", "Right Eye", "Status"]
    ).reset_index().rename(columns={"index":"Frame"})

    report_data = generate_report(df)

    st.subheader("📊 Trip Summary")

    a, b, c = st.columns(3)
    a.metric("Trip Duration (mins)", report_data["total_duration_minutes"])
    b.metric("Drowsy %", f"{report_data['drowsy_percentage']}%")
    c.metric("Total Yawns", report_data["total_yawns"])

    #### ✅ ALWAYS SHOWS GRAPHS (Streams built-in)
    st.write("### 👁️ EAR Trend (Eye Aspect Ratio)")
    st.line_chart(df["EAR"])

    st.write("### 👄 MAR Trend (Mouth Aspect Ratio)")
    st.line_chart(df["MAR"])

    st.write("### 🧠 Awake vs Drowsy Frames")
    st.bar_chart(df["Status"].value_counts())

    #### ✅ AI SAFETY MESSAGE
    insight = get_gemini_summary(report_data)
    st.subheader("🧠 AI Driving Insight")
    st.info(insight)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            file_name=f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "📥 Download JSON",
            json.dumps(report_data, indent=4),
            file_name=f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if clear:
    st.session_state.trip_data = []
    st.info("Data cleared.")

if st.session_state.get("trip_running", False):
    for frame, EAR, MAR, yawn, alarm, left_pred, right_pred in run_drowsiness_detection():
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        status = "⚠️ DROWSY!" if alarm else "✅ Awake"
        color = "red" if alarm else "green"
        status_placeholder.markdown(
            f"### Status: <span style='color:{color}'>{status}</span>",
            unsafe_allow_html=True
        )

        stats_placeholder.write({
            "EAR": round(EAR, 3),
            "MAR": round(MAR, 3),
            "Yawns": yawn,
            "Left Eye": left_pred,
            "Right Eye": right_pred
        })

        st.session_state.trip_data.append([
            EAR, MAR, yawn, left_pred, right_pred,
            "Drowsy" if alarm else "Awake"
        ])
