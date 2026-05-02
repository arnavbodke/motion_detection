# Motion Detection System

A real time motion detection system built using OpenCV and Streamlit.

## Features
- Real time motion detection (webcam & video)
- ROI (Region of Interest) support
- Motion triggered image capture
- Motion triggered video recording
- Streamlit based interactive UI
- Download captured frames

## Tech Stack
- Python
- OpenCV
- Streamlit
- NumPy

## How it works
The system compares consecutive frames using frame differencing. If significant changes are detected, contours are identified and motion is highlighted.
