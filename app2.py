import os
# Tắt hỗ trợ phần cứng GPU để tránh lỗi trên server Linux
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import tempfile
import av

# ==========================================
# 1. CẤU HÌNH VÀ TIỆN ÍCH
# ==========================================
st.set_page_config(page_title="BioGuard Pro - AI Analysis", layout="wide", page_icon="🏃‍♂️")

def calculate_angle(a, b, c):
    """Tính góc giữa 3 điểm (Hông, Gối, Cổ chân)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

def get_status_color(angle):
    """Trả về trạng thái, màu sắc dựa trên góc gập gối"""
    if angle < 90:
        return "NGUY HIEM: Gap goi qua gat!", (0, 0, 255)  # Đỏ
    elif angle < 120:
        return "CANH BAO: Chiu tai lon", (0, 255, 255)   # Vàng
    else:
        return "AN TOAN", (0, 255, 0)                     # Xanh

# Cấu hình STUN servers của Google
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# ==========================================
# 2. LỚP XỬ LÝ VIDEO CAMERA (WEBRTC)
# ==========================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.target_leg = "Chân Trái"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Xử lý MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Xác định index landmark
                if self.target_leg == "Chân Trái":
                    idx = [self.mp_pose.PoseLandmark.LEFT_HIP.value, 
                           self.mp_pose.PoseLandmark.LEFT_KNEE.value, 
                           self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                else:
                    idx = [self.mp_pose.PoseLandmark.RIGHT_HIP.value, 
                           self.mp_pose.PoseLandmark.RIGHT_KNEE.value, 
                           self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                # Tọa độ normalized và pixel
                points = [[landmarks[i].x, landmarks[i].y] for i in idx]
                angle = calculate_angle(points[0], points[1], points[2])
                status, color = get_status_color(angle)
                
                # Vẽ cảnh báo
                knee_px = tuple(np.multiply(points[1], [w, h]).astype(int))
                cv2.putText(img, f"Goc: {angle} deg", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if angle < 90:
                    cv2.circle(img, knee_px, 15, color, -1)

                # Vẽ bộ khung xương
                self.mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            except Exception: pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (ST)
# ==========================================
st.title("🏃‍♂️ BioGuard Pro: AI Biomechanical Analysis")
st.markdown("Hệ thống giám sát vận động thời gian thực giúp ngăn ngừa chấn thương khớp gối.")

st.sidebar.header("⚙️ Cấu hình")
source = st.sidebar.radio("Nguồn dữ liệu:", ("Camera Trực Tiếp", "Tải Video lên"))
leg = st.sidebar.selectbox("Chân theo dõi:", ("Chân Trái", "Chân Phải"))

if source == "Camera Trực Tiếp":
    st.info("Đang sử dụng WebRTC. Nhấn START để kích hoạt camera.")
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
    file = st.sidebar.file_uploader("Tải video (MP4, MOV):", type=['mp4', 'mov'])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        if st.sidebar.button("Bắt đầu Phân tích", type="primary"):
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=0) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    
                    if res.pose_landmarks:
                        lms = res.pose_landmarks.landmark
                        try:
                            # Logic giống hệt recv()
                            if leg == "Chân Trái":
                                p_idx = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value]
                            else:
                                p_idx = [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                            
                            pts = [[lms[i].x, lms[i].y] for i in p_idx]
                            ang = calculate_angle(pts[0], pts[1], pts[2])
                            txt, clr = get_status_color(ang)
                            
                            # Vẽ lên ảnh (Lưu ý vẽ bằng BGR rồi chuyển RGB để hiện Streamlit)
                            cv2.putText(frame, f"Goc: {ang} deg", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                            cv2.putText(frame, txt, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
                            mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        except: pass
                    
                    stframe.image(frame, channels="BGR", use_column_width=True)
            cap.release()
