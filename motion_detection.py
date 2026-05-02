import cv2
import time
import os


def detect_motion(frame, prev_gray, min_area=20):
    motion_detected = False

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

    diff = cv2.absdiff(prev_gray, current_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, current_gray, motion_detected


def save_frame(frame, save_path):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_path, f"motion_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def draw_roi(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame