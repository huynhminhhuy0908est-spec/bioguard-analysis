import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import time
import mediapipe as mp

# ==========================================
# 1. CẤU HÌNH TRANG STREAMLIT VÀ HÀM PHỤ TRỢ
# ==========================================
st.set_page_config(page_title="BioGuard Pro - AI Injury Analysis", layout="wide", page_icon="🏃‍♂️")

# Hàm tính góc giữa 3 điểm
def calculate_angle(a, b, c):
    a = np.array(a) # Hông
    b = np.array(b) # Đầu gối
    c = np.array(c) # Mắt cá chân
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return int(angle)

# ==========================================
# 2. GIAO DIỆN NGƯỜI DÙNG (UI DASHBOARD)
# ==========================================
st.title("🏃‍♂️ BioGuard Pro: Biomechanical Analysis")
st.markdown("Hệ thống AI theo dõi biên độ vận động và cảnh báo rủi ro chấn thương khớp gối.")

# Thanh bên (Sidebar) cho các tùy chọn cài đặt
st.sidebar.header("⚙️ Cài đặt hệ thống")
input_source = st.sidebar.radio("Nguồn Video:", ("Webcam trực tiếp", "Tải Video lên"))
target_leg = st.sidebar.radio("Chân cần phân tích:", ("Chân Trái", "Chân Phải"))

# Upload video nếu người dùng chọn
video_file = None
if input_source == "Tải Video lên":
    video_file = st.sidebar.file_uploader("Chọn file video (mp4, mov)", type=['mp4', 'mov'])

start_btn = st.sidebar.button("Bắt đầu Phân tích", type="primary")
stop_btn = st.sidebar.button("Dừng hệ thống")

# Chia bố cục màn hình chính
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Luồng Video AI")
    # Khung chứa video
    frame_placeholder = st.empty()

with col2:
    st.subheader("Trạng thái Cơ sinh học")
    angle_metric = st.empty()
    status_alert = st.empty()
    advice_text = st.empty()

# ==========================================
# 3. LÕI XỬ LÝ AI & COMPUTER VISION
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Logic chạy khi bấm nút "Bắt đầu"
if start_btn and not stop_btn:
    # Xác định nguồn video
    if input_source == "Webcam trực tiếp":
        cap = cv2.VideoCapture(0)
    else:
        if video_file is not None:
            # Lưu file tạm để OpenCV có thể đọc
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.error("Vui lòng tải video lên trước khi bắt đầu!")
            st.stop()

    # Khởi tạo mô hình Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.info("Đã kết thúc video hoặc mất kết nối camera.")
                break

            # Tối ưu hiệu suất và chuyển đổi hệ màu
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            # Khởi tạo biến góc mặc định
            angle = 0
            
            # Xử lý tọa độ nếu nhận diện được cơ thể
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # Lấy tọa độ tùy theo chân được chọn
                    if target_leg == "Chân Trái":
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    else:
                        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    # Tính toán góc
                    angle = calculate_angle(hip, knee, ankle)
                    
                    # Cập nhật thông số lên Dashboard
                    angle_metric.metric(label=f"Góc Gập {target_leg} (Độ)", value=f"{angle}°")
                    
                    # Logic Cảnh báo rủi ro (Giả định góc < 90 độ khi tiếp đất là nguy hiểm)
                    if angle < 90:
                        status_alert.error("🚨 NGUY HIỂM: Gập gối quá gắt!")
                        advice_text.warning("Nguy cơ cao: Quá tải Dây chằng chéo trước (ACL). Hãy giảm tốc độ tiếp đất.")
                        # Vẽ cảnh báo trực tiếp lên video
                        cv2.circle(image, tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 15, (255, 0, 0), -1)
                    elif angle < 120:
                        status_alert.warning("⚠️ CẢNH BÁO: Đang chịu tải lớn.")
                        advice_text.info("Trạng thái căng cơ. Theo dõi quỹ đạo tiếp đất.")
                    else:
                        status_alert.success("✅ AN TOÀN: Biên độ bình thường.")
                        advice_text.write("Tư thế ổn định, góc gối an toàn.")
                        
                except Exception as e:
                    pass

                # Vẽ khung xương lên video (Tùy chỉnh màu sắc)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Điểm khớp
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Đường nối
                )
            
            # Đưa khung hình đã xử lý lên giao diện Streamlit
            frame_placeholder.image(image, channels="RGB", use_column_width=True)
            
            # Giữ frame rate ổn định (khoảng 30 FPS)
            time.sleep(0.03)

    cap.release()