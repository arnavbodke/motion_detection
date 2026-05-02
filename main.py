import cv2
import os
import time
from motion_detection import detect_motion, save_frame

save_path = "motion_images"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
if not ret:
    print("Error accessing camera")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

last_saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, prev_gray, motion_detected = detect_motion(frame, prev_gray)

    if motion_detected and time.time() - last_saved > 2:
        filename = save_frame(frame, save_path)
        print(f"Saved: {filename}")
        last_saved = time.time()

    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()