import numpy as np
from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import os
import mysql.connector
from modules.sort import Sort


def car_detection(video_path, file_path, line_position=300):
    """
    Detects and tracks cars in a video using YOLOv8 and SORT tracker.
    Logs detected cars to a new Excel file and a MySQL database for each run.
    Shows detection line in the output video.

    Args:
        video_path (str): Path to the input video file.
        file_path (str): Path to the Excel file for logging.
        line_position (int): Y-coordinate of the detection line.

    Returns:
        detection_data (list): List of dictionaries containing detection details.
        processed_video_path (str): Path to the processed video file.
    """
    # Initialize YOLO model
    model = YOLO('../models/yolov8l.pt')

    # MySQL connection setup
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="Abhishek",
            password="Abhi@9324",
            database="object_detection"
        )
        cursor = conn.cursor()
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return [], None

    # Initialize SORT tracker
    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    # Open video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return [], None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define line limits
    limits = [0, line_position, frame_width, line_position]

    # Generate unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_video_path = f"processed_video_{timestamp}.mp4"

    # Create VideoWriter - using alternative fourcc specification
    fourcc = 0x7634706d  # This is 'mp4v' in hexadecimal
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    logged_ids = set()
    detection_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw detection line (green color)
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Detection Line", (frame_width - 150, limits[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # YOLO detection
        results = model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                if class_name in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        # Update tracker
        results_tracker = tracker.update(detections)

        # Process tracked objects
        for result in results_tracker:
            x1, y1, x2, y2, object_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Calculate centroid
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"ID: {int(object_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Log if centroid crosses the line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if object_id not in logged_ids:
                    logged_ids.add(object_id)
                    detection_data.append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Object_ID": int(object_id),
                        "Object_Type": "car"
                    })
                    # Log to MySQL
                    sql = "INSERT INTO detections (timestamp, object_id, object_type) VALUES (NOW(), %s, %s)"
                    values = (int(object_id), "car")
                    cursor.execute(sql, values)
                    conn.commit()

        # Write the processed frame to the output video
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    cursor.close()
    conn.close()
    print("Video processing complete. Resources released.")

    # Create fresh Excel file with only current detection data
    df = pd.DataFrame(detection_data)
    df.to_excel(file_path, index=False)

    return detection_data, processed_video_path