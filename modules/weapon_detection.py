import cv2
import pandas as pd
import datetime
from ultralytics import YOLO
import os

# Load YOLOv7 model
model_path = "../models/weaponcustom.pt"
model = YOLO(model_path)

# Results directory
RESULTS_DIR = "static/results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Labels considered as weapons
anomalous_objects = ["knife", "gun", "pistol", "weapon"]

def enhance_night_vision(frame):
    """Apply CLAHE to enhance low-light images."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

def resize_to_fit(img, width=1200, height=800):
    """Resize image to fit within given width and height."""
    h, w = img.shape[:2]
    scaling_factor = min(width / w, height / h)
    new_size = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def process_image(image_path):
    """Detect weapons in an image, save results with bounding boxes, and log to Excel."""
    log_data = []
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Error reading image: {image_path}")

    frame = cv2.resize(frame, (1280, 1040))
    enhanced_frame = enhance_night_vision(frame)
    results = model(enhanced_frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            label = result.names[int(box.cls[0])]
            color = (0, 0, 255) if label in anomalous_objects else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Log if it's a weapon
            if label in anomalous_objects:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_data.append([timestamp, label, confidence])

    # Save processed image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_weapon_{timestamp}.jpg"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    cv2.imwrite(output_path, frame)

    # Save to Excel
    excel_path = None
    if log_data:
        excel_path = os.path.join(RESULTS_DIR, f"weapon_detections_{timestamp}.xlsx")
        df = pd.DataFrame(log_data, columns=["Timestamp", "Object", "Confidence"])
        df.to_excel(excel_path, index=False)

    return output_filename, excel_path

def process_video(video_path):
    """Detect weapons in a video, save annotated video, and log detections to Excel."""
    log_data = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"processed_weapon_{timestamp}.mp4"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    excel_filename = f"weapon_detections_{timestamp}.xlsx"
    excel_path = os.path.join(RESULTS_DIR, excel_filename)

    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 1040))
        enhanced_frame = enhance_night_vision(frame)
        results = model(enhanced_frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)
                label = result.names[int(box.cls[0])]
                color = (0, 0, 255) if label in anomalous_objects else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label in anomalous_objects:
                    frame_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_data.append([frame_time, label, confidence])

        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        out.write(frame)

    cap.release()
    if out:
        out.release()

    if log_data:
        df = pd.DataFrame(log_data, columns=["Timestamp", "Object", "Confidence"])
        df.to_excel(excel_path, index=False)

    return output_filename, excel_path if log_data else (output_filename, None)
