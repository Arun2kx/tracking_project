# Multi-Object Detection and Tracking

## Overview

This project implements a computer vision pipeline to detect and track multiple players in a sports video. Each player is assigned a unique ID and tracked across frames.

## Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* DeepSORT

## How It Works

* YOLOv8 detects players in each frame
* DeepSORT assigns and maintains unique IDs
* The output video shows bounding boxes and IDs

## How to Run

1. Install dependencies:
   pip install ultralytics opencv-python deep-sort-realtime

2. Place input video as:
   input.mp4

3. Run:
   python main.py

## Output

* output.mp4 with tracking and IDs

## Challenges

* Occlusion (players overlap)
* Similar-looking players

## Improvements

* Use GPU for faster processing
* Improve tracking with re-identification models

## Video Source

(Add your video link here)
