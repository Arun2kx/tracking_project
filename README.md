# Multi-Object Detection and Tracking using YOLOv8 + DeepSORT

## 📌 Overview

This project implements a computer vision pipeline to detect and track multiple players in a sports video. Each player is assigned a unique ID and tracked consistently across frames.

## ⚙️ Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* DeepSORT

## 🚀 How It Works

* YOLOv8 detects players in each frame
* DeepSORT assigns unique IDs and maintains identity across frames
* Bounding boxes and IDs are drawn on the output video

## ▶️ How to Run

1. Install dependencies:
   pip install ultralytics opencv-python deep-sort-realtime

2. Place input video as:
   input.mp4

3. Run:
   python main.py

## 📊 Output

* output.mp4 with bounding boxes and unique IDs for each player

## ⚠️ Challenges

* Occlusion (players overlapping)
* Similar-looking players causing ID switching

## 🔧 Improvements

* Use GPU for faster processing
* Improve tracking using re-identification models

## 🔗 Video Source

(Add your video link here)

## 📌 Note

Due to GitHub file size limitations, video files are not uploaded. Output video can be shared via Google Drive.
