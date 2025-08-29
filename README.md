
# 🚗 Driver Safety Detection Project  

This project detects **distracted or risky driving behaviors** using a trained YOLO model and provides **real-time alerts** through a Streamlit app.  

The system can identify actions such as:  
- 📱 **Texting**  
- 📞 **Talking on the phone**  
- 🥤 **Drinking**  
- 🎵 **Operating the radio**  
- 😴 **Drowsiness (eye closing / nodding off)**  

When risky behaviors last longer than a set threshold, the system triggers a **siren alarm** 🚨 and highlights the detected area with a blinking box.  

---

## 🛠️ Features  

### 🔹 Streamlit Application  
- Upload and analyze driving videos in real-time.  
- Detect risky behaviors using **YOLO11n** lightweight model.  
- Adjustable **time thresholds** per behavior (e.g., shorter for drowsiness, longer for drinking).  
- **Blinking bounding boxes** (Red/Yellow) on detected risky actions.  
- Continuous **siren alarm** until the driver corrects their behavior or the user presses stop.  
- FPS-based timing ensures detection is consistent with video speed.  
- Clear alerts shown **below the video**, without cluttering the frame.  

### 🔹 Training Insights (YOLO Model)  
- **Model:** YOLO11n (fast + lightweight for real-time inference).  
- **Training Hyperparameters:**  
  - Optimizer: `AdamW` (stable + better generalization)  
  - Learning Rate: `1e-3` (with decay factor `0.01`)  
  - Weight Decay: `0.0005` (regularization)  
  - Dropout: `0.2` (prevent overfitting)  
  - Epochs: `50` with **early stopping (patience=10)**  
  - Batch size: `32`, image size: `640x640`  

📊 **Training Results:**  
- High accuracy in risky behavior detection (`mAP50 ~ 98%`).  
- Overfitting reduced with dropout + weight decay.  
- `AdamW` optimizer improved convergence compared to standard SGD.  

## 🚀 How to Run  

1. **Install dependencies**:  
   ```bash
   pip install streamlit ultralytics opencv-python pygame
   
2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py

3.**Upload a video** of driving and watch detection in action

## 🔮 Future Improvements

### 🧠 Lightweight Model Integration
Integrate MobileNet or lightweight CNNs for on-device real-time classification.

### 📷 Live Webcam Detection
Add webcam live detection mode for real-world testing.

### 🎙 Custom Voice Alerts
Replace siren with custom voice alerts ("⚠️ Please stay alert!").

### ⏱ Adaptive Thresholds
Implement adaptive thresholds (shorter at night for drowsiness, longer during day).

### 🌍 Mobile Deployment
Deploy as a mobile app for Android Auto / car dashboards.

### 🛠 Advanced Face Analysis
Add driver face landmarks (eye blink, head pose) for deeper drowsiness detection.

## 📌 Conclusion

This project showcases the fusion of deep learning and computer vision to enhance driver safety. The system reliably detects distracted behaviors, and the Streamlit interface provides an intuitive platform for demonstration and testing. Future work will focus on real-time optimization and expanded feature integration.