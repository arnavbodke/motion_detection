import streamlit as st
import cv2
import os
import time
from motion_detection import detect_motion, save_frame, draw_roi

st.set_page_config(layout="wide")
st.title("Motion Detection System")

mode = st.sidebar.radio("Input Source", ["Webcam", "Upload Video"])
save_images = st.sidebar.checkbox("Save Images", True)
record_video = st.sidebar.checkbox("Record Video on Motion", True)
min_area = st.sidebar.slider("Sensitivity", 10, 3000, 50)

use_roi = st.sidebar.checkbox("Enable ROI", False)

start = st.sidebar.button("Start")
stop = st.sidebar.button("Stop")

roi = (0, 0, 640, 480)
if use_roi:
    x = st.sidebar.slider("X", 0, 640, 0)
    y = st.sidebar.slider("Y", 0, 480, 0)
    w = st.sidebar.slider("Width", 50, 640, 300)
    h = st.sidebar.slider("Height", 50, 480, 300)
    roi = (x, y, w, h)

frame_placeholder = st.empty()
status_placeholder = st.empty()
stats_placeholder = st.empty()

save_path = "motion_images"
video_path = "videos"

os.makedirs(save_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)
os.makedirs("temp", exist_ok=True)

cap = None

if mode == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])
    if video_file:
        temp_path = "temp/temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_path)

elif mode == "Webcam" and start:
    cap = cv2.VideoCapture(0)

if cap and start:

    ret, prev_frame = cap.read()

    if not ret:
        st.error("Cannot read video")
    else:

        prev_gray_full = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray_full = cv2.GaussianBlur(prev_gray_full, (5, 5), 0)

        prev_gray_roi = None

        last_saved = 0
        motion_count = 0
        frame_count = 0

        recording = False
        video_writer = None

        while cap.isOpened():

            if stop:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

       
            if use_roi:
                x, y, w, h = roi

                roi_frame = frame[y:y+h, x:x+w]

                current_gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                current_gray_roi = cv2.GaussianBlur(current_gray_roi, (5, 5), 0)

                if prev_gray_roi is None:
                    prev_gray_roi = current_gray_roi

                processed_frame, prev_gray_roi, motion_detected = detect_motion(
                    roi_frame, prev_gray_roi, min_area
                )

                frame[y:y+h, x:x+w] = processed_frame
                frame = draw_roi(frame, roi)

            else:
                frame, prev_gray_full, motion_detected = detect_motion(
                    frame, prev_gray_full, min_area
                )

        
            if motion_detected:
                motion_count += 1
                status_placeholder.markdown("### 🟢 Motion Detected")

                if save_images and time.time() - last_saved > 2:
                    save_frame(frame, save_path)
                    last_saved = time.time()

                if record_video and not recording:
                    filename = os.path.join(video_path, f"motion_{int(time.time())}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(
                        filename, fourcc, 20.0,
                        (frame.shape[1], frame.shape[0])
                    )
                    recording = True

            else:
                status_placeholder.markdown("### 🔴 No Motion")

                if recording:
                    video_writer.release()
                    recording = False

    
            if recording and video_writer:
                video_writer.write(frame)

    
            frame_placeholder.image(frame, channels="BGR")


            stats_placeholder.markdown(f"""
            **Frames:** {frame_count}  
            **Motion Events:** {motion_count}  
            """)

        cap.release()
        if video_writer:
            video_writer.release()