import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import tempfile
import av

# Cấu hình trang
st.set_page_config(page_title="BioGuard Pro - AI Injury Analysis", layout="wide", page_icon="🏃‍♂️")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # Khởi tạo model 1 lần duy nhất để tiết kiệm tài nguyên
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.target_leg = "Chân Trái"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Thêm .value để lấy đúng index
                if self.target_leg == "Chân Trái":
                    hip_idx, knee_idx, ankle_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value
                else:
                    hip_idx, knee_idx, ankle_idx = self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
                
                hip = [landmarks[hip_idx].x, landmarks[hip_idx].y]
                knee = [landmarks[knee_idx].x, landmarks[knee_idx].y]
                ankle = [landmarks[ankle_idx].x, landmarks[ankle_idx].y]
                
                angle = calculate_angle(hip, knee, ankle)
                
                # Hiển thị thông tin
                h, w, _ = img.shape
                knee_pos = tuple(np.multiply(knee, [w, h]).astype(int))
                
                color, status = (0, 255, 0), "AN TOAN"
                if angle < 90:
                    color, status = (0, 0, 255), "NGUY HIEM: Gap goi qua gat!"
                    cv2.circle(img, knee_pos, 15, color, -1)
                elif angle < 120:
                    color, status = (0, 255, 255), "CANH BAO: Chiu tai lon"

                cv2.putText(img, f"Goc: {angle} do", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            except Exception:
                pass

            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI
st.title("🏃‍♂️ BioGuard Pro: Biomechanical Analysis")
input_source = st.sidebar.radio("Nguồn Video:", ("Camera Trực Tiếp", "Tải Video lên"))
target_leg_choice = st.sidebar.selectbox("Chân cần phân tích:", ("Chân Trái", "Chân Phải"))

if input_source == "Camera Trực Tiếp":
    webrtc_ctx = webrtc_streamer(
        key="bioguard-camera",
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.target_leg = target_leg_choice

else:
    video_file = st.sidebar.file_uploader("Chọn file video (mp4, mov)", type=['mp4', 'mov'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        if st.sidebar.button("Bắt đầu Phân tích", type="primary"):
            # Đồng bộ cách gọi Pose để tránh lỗi
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=0) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        # Tương tự như trên, thêm .value
                        idx = mp_pose.PoseLandmark.LEFT_HIP.value if target_leg_choice == "Chân Trái" else mp_pose.PoseLandmark.RIGHT_HIP.value
                        # ... (Tính toán tương tự recv để đồng nhất kết quả)
                        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    stframe.image(image, channels="RGB", use_column_width=True)
            cap.release()
