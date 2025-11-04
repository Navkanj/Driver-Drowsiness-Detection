import streamlit as st
import cv2
import numpy as np
import pandas as pd
from detect_drowsiness import run_drowsiness_detection
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

def generate_report(df):
    """Generate analysis report from trip data"""
    total_time = len(df) / 30  # Assuming 30 FPS
    drowsy_periods = df[df['Status'] == 'Drowsy']
    alert_periods = df[df['Status'] == 'Awake']
    
    # Convert numpy values to native Python types
    report = {
        "trip_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_minutes": float(round(total_time / 60, 2)),
        "drowsy_percentage": float(round(len(drowsy_periods) / len(df) * 100, 2)),
        "total_yawns": int(df['Yawn Count'].max()),
        "average_ear": float(round(df['EAR'].mean(), 3)),
        "average_mar": float(round(df['MAR'].mean(), 3)),
        "drowsy_episodes": int(len(drowsy_periods['Status'].value_counts())),
    }
    
    # Create visualizations
    fig_ear = px.line(df, y='EAR', title='Eye Aspect Ratio Over Time')
    fig_mar = px.line(df, y='MAR', title='Mouth Aspect Ratio Over Time')
    fig_status = px.pie(df['Status'].value_counts().reset_index(), 
                       values='count', names='Status', 
                       title='Alertness Distribution')
    
    return report, [fig_ear, fig_mar, fig_status]

def save_report(df, report_data):
    """Save report locally and send analytics to Google API"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/trip_report_{timestamp}"
    
    # Convert DataFrame to JSON serializable format
    df_json = df.copy()
    for col in df_json.select_dtypes(include=['int64', 'float64']).columns:
        df_json[col] = df_json[col].astype(float)
    
    # Save locally
    df.to_csv(f"{report_path}.csv", index=False)
    with open(f"{report_path}.json", 'w') as f:
        json.dump(report_data, f, indent=4, default=str)
    
    # Send analytics to Google API (example endpoint)
    try:
        api_endpoint = f"https://api.google.com/v1/analytics?key={GOOGLE_API_KEY}"
        response = requests.post(api_endpoint, json=report_data)
        if response.status_code == 200:
            return True
        else:
            st.warning("Analytics data couldn't be sent, but report saved locally.")
            return True
    except Exception as e:
        st.error(f"Failed to send analytics: {str(e)}")
        return True  # Still return True as local save worked

# Streamlit UI
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("**Hybrid CNN + Mediapipe-based Real-Time Detection**")

# Initialize session state
if 'trip_data' not in st.session_state:
    st.session_state.trip_data = []

FRAME_WINDOW = st.image([])
status_placeholder = st.empty()
stats_placeholder = st.empty()

col1, col2, col3 = st.columns([1,1,1])
with col1:
    start = st.button("▶️ Start Trip", use_container_width=True)
with col2:
    stop = st.button("⏹️ Stop Trip", use_container_width=True)
with col3:
    clear = st.button("🗑️ Clear Data", use_container_width=True)

if start:
    st.session_state.trip_running = True
    st.session_state.trip_data = []
    st.success("Trip Started. Monitoring driver alertness...")

if stop and len(st.session_state.trip_data) > 0:
    st.session_state.trip_running = False
    
    # Generate and display report
    df = pd.DataFrame(st.session_state.trip_data, 
                     columns=["EAR", "MAR", "Yawn Count", "Left Eye", "Right Eye", "Status"])
    
    report_data, figures = generate_report(df)
    
    # Display report sections
    st.subheader("📊 Trip Summary Report")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Trip Duration (mins)", report_data["total_duration_minutes"])
    with metrics_col2:
        st.metric("Drowsy Percentage", f"{report_data['drowsy_percentage']}%")
    with metrics_col3:
        st.metric("Total Yawns", report_data["total_yawns"])
    
    # Display graphs
    for fig in figures:
        st.plotly_chart(fig, use_container_width=True)
    
    # Save report locally and send analytics
    if save_report(df, report_data):
        st.success("Trip report saved to reports folder!")
    
    # Add download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Report CSV",
            data=df.to_csv(index=False),
            file_name=f"trip_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="📥 Download Report JSON",
            data=json.dumps(report_data, indent=4),
            file_name=f"trip_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if clear:
    st.session_state.trip_data = []
    st.info("Trip data cleared.")

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

        st.session_state.trip_data.append([EAR_avg, MAR, yawn_count, left_pred, right_pred, 
                                         "Drowsy" if alarm_on else "Awake"])
