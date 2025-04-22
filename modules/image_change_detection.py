import cv2
import numpy as np
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import mysql.connector
from ultralytics import YOLO


def process_image_changes(before_image_path, after_image_path, diff_image_path, excel_file_path):
    """
    Process two images to detect changes between them and generate a difference image.

    Args:
        before_image_path (str): Path to the 'before' image
        after_image_path (str): Path to the 'after' image
        diff_image_path (str): Path to save the difference image
        excel_file_path (str): Path to save the Excel report

    Returns:
        dict: Dictionary containing processing results
    """
    # Initialize YOLO model
    model = YOLO("../models/yolov8l.pt")

    # Database Connection
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="Abhishek",
        password="Abhi@9324",
        database="object_detection"
    )
    cursor = conn.cursor()

    # Extract filenames
    before_filename = os.path.basename(before_image_path)
    after_filename = os.path.basename(after_image_path)

    # MySQL Logging
    def insert_change(video_name, object_type, change_type, distance_x, distance_y, direction_x, direction_y,
                      color_change=None):
        sql = """
        INSERT INTO image_changes (timestamp, video_name, object_type, change_type, distance_x, distance_y, direction_x, direction_y, color_change) 
        VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (video_name, object_type, change_type, distance_x, distance_y, direction_x, direction_y, color_change)
        cursor.execute(sql, values)
        conn.commit()

    # Load and preprocess images
    def load_and_preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        return image

    before = load_and_preprocess_image(before_image_path)
    after = load_and_preprocess_image(after_image_path)

    # Convert images to grayscale and compute SSIM
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(gray_before, gray_after, full=True)
    diff = (diff * 255).astype("uint8")

    # Detect objects using YOLO
    before_results = model(before)
    after_results = model(after)

    # Generate difference image
    diff_colored = cv2.absdiff(before, after)
    cv2.imwrite(diff_image_path, diff_colored)

    # Extract object positions
    def extract_objects(yolo_results, image):
        objects = []
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names.get(class_id, f"Unknown_{class_id}")
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                objects.append((center_x, center_y, class_name))
        return objects

    before_objects = extract_objects(before_results, before)
    after_objects = extract_objects(after_results, after)

    # Detect changes
    changes_detected = []
    for bcx, bcy, object_name in before_objects:
        found = False
        for acx, acy, after_name in after_objects:
            if object_name == after_name:
                dx, dy = acx - bcx, acy - bcy
                if abs(dx) > 10 or abs(dy) > 10:
                    direction_x = "Right" if dx > 0 else "Left"
                    direction_y = "Down" if dy > 0 else "Up"
                    total_distance = round(np.sqrt(dx**2 + dy**2), 2)  # Euclidean distance

                    insert_change(before_filename, object_name, "Moved", dx, dy, direction_x, direction_y)
                    changes_detected.append((object_name, "Moved", dx, dy, direction_x, direction_y, total_distance))

                found = True
                break
        if not found:
            insert_change(before_filename, object_name, "Removed", 0, 0, "None", "None")
            changes_detected.append((object_name, "Removed", 0, 0, "None", "None", 0))

    for acx, acy, object_name in after_objects:
        if not any(obj_name == object_name for _, _, obj_name in before_objects):
            insert_change(after_filename, object_name, "Added", 0, 0, "None", "None")
            changes_detected.append((object_name, "Added", 0, 0, "None", "None", 0))

    # Save to Excel with additional movement data
    df = pd.DataFrame(
        changes_detected,
        columns=["Object Type", "Change Type", "Distance X", "Distance Y", "Direction X", "Direction Y", "Total Distance"]
    )
    df.to_excel(excel_file_path, index=False)

    # Close MySQL connection
    cursor.close()
    conn.close()

    return {
        "before_image": before_image_path,
        "after_image": after_image_path,
        "diff_image": diff_image_path,
        "excel_path": excel_file_path,
        "changes_detected": len(changes_detected),
        "ssim_score": score
    }
