import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import tempfile
import av

# ==========================================
# 1. CẤU HÌNH TRANG VÀ HÀM PHỤ TRỢ
# ==========================================
st.set_page_config(page_title="BioGuard Pro - AI Injury Analysis", layout="wide", page_icon="🏃‍♂️")

def calculate_angle(a, b, c):
    """Tính góc giữa 3 điểm (Hông, Gối, Cổ chân)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return int(angle)

# Cấu hình STUN server của Google để hỗ trợ kết nối qua nhiều loại mạng
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# ==========================================
# 2. LỚP XỬ LÝ VIDEO WEBRTC THỜI GIAN THỰC
# ==========================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_leg = "Chân Trái"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Xác định các điểm mốc dựa trên chân được chọn
                if self.target_leg == "Chân Trái":
                    hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                else:
                    hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                angle = calculate_angle(hip, knee, ankle)
                
                # Logic hiển thị và cảnh báo
                h, w, _ = img.shape
                knee_pos = tuple(np.multiply(knee, [w, h]).astype(int))
                
                color = (0, 255, 0) # Xanh - An toàn
                status = "AN TOAN"
                
                if angle < 90:
                    color = (0, 0, 255) # Đỏ - Nguy hiểm
                    status = "NGUY HIEM: Gap goi qua gat!"
                    cv2.circle(img, knee_pos, 15, color, -1)
                elif angle < 120:
                    color = (0, 255, 255) # Vàng - Cảnh báo
                    status = "CANH BAO: Chiu tai lon"

                cv2.putText(img, f"Goc: {angle} do", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            except Exception:
                pass

            # Vẽ khung xương
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG
# ==========================================
st.title("🏃‍♂️ BioGuard Pro: Biomechanical Analysis")
st.markdown("Hệ thống AI theo dõi biên độ vận động và cảnh báo rủi ro chấn thương khớp gối.")

st.sidebar.header("⚙️ Cài đặt hệ thống")
input_source = st.sidebar.radio("Nguồn Video:", ("Camera Trực Tiếp", "Tải Video lên"))
target_leg_choice = st.sidebar.selectbox("Chân cần phân tích:", ("Chân Trái", "Chân Phải"))

if input_source == "Camera Trực Tiếp":
    st.info("Nhấn 'START' để cấp quyền truy cập Camera.")
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
        
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        if st.sidebar.button("Bắt đầu Phân tích", type="primary"):
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        try:
                            # Đồng bộ logic lấy tọa độ chân
                            if target_leg_choice == "Chân Trái":
                                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                            else:
                                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                            
                            angle = calculate_angle(hip, knee, ankle)
                            
                            # Vẽ lên ảnh (RGB)
                            color = (0, 255, 0) if angle >= 120 else (255, 255, 0) if angle >= 90 else (255, 0, 0)
                            cv2.putText(image, f"Goc: {angle} do", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        except: pass
                    
                    stframe.image(image, channels="RGB", use_column_width=True)
            cap.release()
