import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import tempfile
import av

# ==========================================
# 1. KHỞI TẠO & CẤU HÌNH
# ==========================================
st.set_page_config(page_title="BioGuard Pro - AI Injury Prevention", layout="wide", page_icon="🏃‍♂️")

# CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #161b22; }
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    </style>
    """, unsafe_allow_stdio=True)

if 'min_angle' not in st.session_state:
    st.session_state.min_angle = 180

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return int(360 - angle if angle > 180.0 else angle)

def get_status(angle):
    if angle < 90: return "NGUY HIỂM: Gập quá gắt!", (0, 0, 255), "🔴"
    if angle < 120: return "CẢNH BÁO: Tải trọng lớn", (0, 255, 255), "🟡"
    return "AN TOÀN", (0, 255, 0), "🟢"

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ==========================================
# 2. XỬ LÝ VIDEO CAMERA (WEBRTC)
# ==========================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_leg = "Chân Trái"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            idx = [self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE] if self.target_leg == "Chân Trái" else \
                  [self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            try:
                pts = [[lms[i.value].x, lms[i.value].y] for i in idx]
                angle = calculate_angle(pts[0], pts[1], pts[2])
                status, color, icon = get_status(angle)
                
                # Vẽ đồ họa
                cv2.putText(img, f"Goc: {angle} deg", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            except: pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
st.title("🏃‍♂️ BioGuard Pro: AI Biomechanical Analysis")

col1, col2 = st.columns([3, 1])

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843032.png", width=100)
    st.header("⚙️ Điều Khiển")
    source = st.radio("Nguồn video:", ("Camera Trực Tiếp", "Tải Video lên"))
    leg = st.selectbox("Chân theo dõi:", ("Chân Trái", "Chân Phải"))
    if st.button("🔄 Reset Chỉ Số"):
        st.session_state.min_angle = 180
        st.rerun()

with col1:
    if source == "Camera Trực Tiếp":
        ctx = webrtc_streamer(
            key="bioguard-v3",
            video_processor_factory=PoseProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True
        )
        if ctx.video_processor:
            ctx.video_processor.target_leg = leg
    else:
        file = st.file_uploader("Tải video tập luyện:", type=['mp4', 'mov'])
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            if st.button("Bắt đầu Phân tích", type="primary"):
                with mp.solutions.pose.Pose(model_complexity=0) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.resize(frame, (640, 480))
                        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if res.pose_landmarks:
                            lms = res.pose_landmarks.landmark
                            p_idx = [23, 25, 27] if leg == "Chân Trái" else [24, 26, 28]
                            pts = [[lms[i].x, lms[i].y] for i in p_idx]
                            ang = calculate_angle(pts[0], pts[1], pts[2])
                            if ang < st.session_state.min_angle: st.session_state.min_angle = ang
                            
                            txt, clr, _ = get_status(ang)
                            cv2.putText(frame, f"Goc: {ang}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
                            mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                        stframe.image(frame, channels="BGR")
            cap.release()

with col2:
    st.subheader("📊 Chỉ số Session")
    status_msg, _, icon = get_status(st.session_state.min_angle)
    st.metric("Góc gập sâu nhất", f"{st.session_state.
