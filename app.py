import streamlit as st
import cv2
import tempfile
import time
import threading
from collections import defaultdict
from ultralytics import YOLO
import pygame
import pandas as pd

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load siren sound
SIREN_FILE = "siren.wav"
try:
    siren = pygame.mixer.Sound(SIREN_FILE)
except:
    st.error("❌ Siren sound file not found. Please place siren.wav in your project folder.")

# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# --- Realistic thresholds for academic demo ---
SIREN_THRESHOLDS = {
    "Eyes Closed": 3.0,
    "Nodding Off": 3.0,
    "Texting": 8.0,
    "Talking on the phone": 8.0,
    "Yawning": 8.0,
    "Drinking": 8.0,
    "Operating the Radio": 8.0,
    "Reaching Behind": 8.0,
    "Hair and Makeup": 8.0,
    "Talking to Passenger": 8.0,
}

# --- Alarm Controls ---
audio_active = False
alarm_active = False
alarm_volume = 0.2
siren.set_volume(alarm_volume)

def play_siren():
    global audio_active, alarm_volume
    while audio_active:
        siren.play()
        if alarm_volume < 1.0:
            alarm_volume = min(1.0, alarm_volume + 0.05)
            siren.set_volume(alarm_volume)
        time.sleep(0.5)

def start_alarm():
    global audio_active, alarm_active, alarm_volume
    if not audio_active:
        audio_active = True
        alarm_active = True
        alarm_volume = 0.2
        siren.set_volume(alarm_volume)
        threading.Thread(target=play_siren, daemon=True).start()

def stop_alarm():
    global audio_active, alarm_active
    audio_active = False
    alarm_active = False
    siren.stop()

# --- App UI ---
st.title("🚗 Driver Safety Detector (YOLO11)")
st.markdown(
    "Upload a driver video, and the app will detect unsafe behaviors. "
    "If a risky behavior lasts beyond its threshold, a **siren alarm** will trigger 🚨"
)

uploaded_file = st.file_uploader("Upload Driver Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.03

    active_behaviors = defaultdict(float)
    stframe = st.empty()
    alert_placeholder = st.empty()
    progress_placeholder = st.empty()

    st.markdown("### Video Processing...")

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        risky_detected = False
        risky_class = None
        frame_counter += 1

        for r in results:
            boxes = r.boxes
            names = model.names

            for box in boxes:
                cls_id = int(box.cls[0])
                cls = names[cls_id]

                # Clean class name
                cls_clean = cls.split(" - ")[-1].strip() if " - " in cls else cls
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_clean in SIREN_THRESHOLDS:
                    active_behaviors[cls_clean] += frame_time
                    elapsed = active_behaviors[cls_clean]
                    threshold = SIREN_THRESHOLDS[cls_clean]

                    if elapsed >= threshold:
                        risky_detected = True
                        risky_class = cls_clean

                        # Blinking rectangle every 5 frames
                        alert_red = (frame_counter // 5) % 2 == 0
                        alert_color = (0, 0, 255) if alert_red else (0, 255, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 4)

                        # Start alarm if not already active
                        if not alarm_active:
                            start_alarm()
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                cv2.putText(frame, cls_clean, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Alert panel under video
        if risky_detected:
            elapsed = active_behaviors[risky_class]
            threshold = SIREN_THRESHOLDS[risky_class]
            progress_value = min(1.0, elapsed / threshold)

            alert_placeholder.markdown(
                f"<h2 style='color:red;'>⚠️ ALERT: {risky_class} detected! 🚨</h2>",
                unsafe_allow_html=True
            )
            progress_placeholder.progress(progress_value, text=f"{elapsed:.1f}s / {threshold:.1f}s")

        else:
            alert_placeholder.empty()
            progress_placeholder.empty()
            stop_alarm()

        stframe.image(frame, channels="BGR")

    cap.release()
    stop_alarm()

    # --- Summary Table with Alarm Highlight ---
    summary_data = []
    for behavior, duration in active_behaviors.items():
        threshold = SIREN_THRESHOLDS.get(behavior, 0)
        alarm_triggered = "Yes" if duration >= threshold else "No"
        summary_data.append({
            "Behavior": behavior,
            "Total Duration (s)": round(duration, 2),
            "Alarm Triggered": alarm_triggered
        })

    df_summary = pd.DataFrame(summary_data)

    # Highlight rows in red where alarm triggered
    def highlight_alarm(row):
        return ['background-color: #ffcccc' if row['Alarm Triggered'] == 'Yes' else '' for _ in row]

    st.markdown("### 📊 Detected Risky Behaviors Summary")
    st.dataframe(df_summary.style.apply(highlight_alarm, axis=1))

    st.success("✅ Video processed successfully!")













