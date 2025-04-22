import cv2
import os
import pandas as pd  # ✅ For Excel logging
from ultralytics import YOLO
from datetime import datetime

# Load the YOLO model trained for low-light detection
model = YOLO("../models/yolov8xcdark.pt")  # Ensure correct model path

# Ensure results directory exists
RESULTS_DIR = "static/results/"
os.makedirs(RESULTS_DIR, exist_ok=True)


def resize_to_fit(img, width=1200, height=800):
    """Resize image while maintaining aspect ratio."""
    h, w = img.shape[:2]
    scaling_factor = min(width / w, height / h)
    new_size = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def process_image(image_path):
    """Process an image for night vision object detection and log details to Excel."""
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        return None, None

    results = model(img)  # Run YOLO detection
    img = results[0].plot()  # Draw bounding boxes
    img = resize_to_fit(img)  # Resize

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}.jpg"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    cv2.imwrite(output_path, img)
    print(f"Processed image saved at: {output_path}")

    # ✅ Save object details to Excel
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append([
                timestamp,  # Timestamp
                r.names[int(box.cls)],  # Object name
                float(box.conf)  # Confidence score
            ])

    excel_filename = f"detections_{timestamp}.xlsx"
    excel_path = os.path.join(RESULTS_DIR, excel_filename)

    if detections:
        df = pd.DataFrame(detections, columns=["Timestamp", "Object", "Confidence"])
        df.to_excel(excel_path, index=False)
        print(f"Detection log saved at: {excel_path}")

    return output_filename, excel_filename  # ✅ Return both filenames


def process_video(video_path):
    """Process a video frame by frame for night vision detection, save output video, and log details to Excel."""
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' not found.")
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None, None

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}.mp4"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    excel_filename = f"detections_{timestamp}.xlsx"
    excel_path = os.path.join(RESULTS_DIR, excel_filename)

    # Define video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # List to store detections
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        results = model(frame)  # Run detection
        img = results[0].plot()  # Draw bounding boxes
        img = resize_to_fit(img, width, height)  # Resize frame

        out.write(img)  # Save processed frame

        # ✅ Save object details with timestamp
        frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            for box in r.boxes:
                detections.append([
                    frame_time,  # Timestamp
                    r.names[int(box.cls)],  # Object name
                    float(box.conf)  # Confidence score
                ])

    cap.release()
    out.release()

    print(f"Processed video saved at: {output_path}")

    # ✅ Save detections to Excel
    if detections:
        df = pd.DataFrame(detections, columns=["Timestamp", "Object", "Confidence"])
        df.to_excel(excel_path, index=False)
        print(f"Detection log saved at: {excel_path}")

    return output_filename, excel_filename  # ✅ Return both video and Excel filenames
