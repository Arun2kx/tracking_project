import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture("input.mp4")

# Check video opened
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Get properties safely
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25  # fallback

# Create VideoWriter
out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (important for stability)
    frame = cv2.resize(frame, (width, height))

    # Detection
    results = model(frame)[0]

    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r

        if int(class_id) == 0:
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'person'))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        cv2.rectangle(frame, (int(l), int(t)),
                      (int(l + w), int(t + h)),
                      (0, 255, 0), 2)

        cv2.putText(frame, f"ID: {track_id}",
                    (int(l), int(t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # WRITE FRAME (important)
    out.write(frame)

# Release
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output video saved as output.mp4")