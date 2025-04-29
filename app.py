from flask import Flask, render_template, request, send_file, redirect, url_for, send_from_directory,abort
import os
from datetime import datetime

from modules.car_detection import car_detection
from modules.image_change_detection import process_image_changes
# Import modules with aliases
import modules.night as night_module
import modules.weapon_detection as weapon_module



app = Flask(__name__)

# Disable watching Python library files
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/car-detection", methods=["GET", "POST"])
def car_detection():
    if request.method == "POST":
        uploaded_file = request.files.get("video")
        if uploaded_file and uploaded_file.filename != "":
            try:
                temp_path = os.path.join("static", "uploads", uploaded_file.filename)
                uploaded_file.save(temp_path)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_path = f"detections_{timestamp}.xlsx"
                output_video = f"processed_{timestamp}.mp4"

                from modules.car_detection import car_detection as process_video
                detection_data, video_path = process_video(
                    video_path=temp_path,
                    file_path=excel_path
                )

                return render_template(
                    "car_detection.html",
                    processed_video_path=video_path,
                    excel_file_path=excel_path,
                    processing_message="Processing complete!"
                )

            except Exception as e:
                return render_template("car_detection.html", error_message=str(e))

    return render_template("car_detection.html")

@app.route("/image-change", methods=["GET", "POST"])
def image_change():
    if request.method == "POST":
        before_image = request.files["before_image"]
        after_image = request.files["after_image"]

        if before_image and after_image:
            try:
                os.makedirs("static/uploads", exist_ok=True)
                os.makedirs("static/results", exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                before_filename = f"before_{timestamp}.jpg"
                after_filename = f"after_{timestamp}.jpg"
                diff_filename = f"diff_{timestamp}.jpg"
                excel_filename = f"changes_{timestamp}.xlsx"

                before_path = os.path.join("static", "uploads", before_filename)
                after_path = os.path.join("static", "uploads", after_filename)
                diff_image_path = os.path.join("static", "results", diff_filename)

                before_image.save(before_path)
                after_image.save(after_path)

                result = process_image_changes(before_path, after_path, diff_image_path, excel_filename)

                return render_template(
                    "image_change.html",
                    before_image=f"uploads/{before_filename}",
                    after_image=f"uploads/{after_filename}",
                    diff_image=f"results/{diff_filename}",
                    excel_file_path=result["excel_path"],
                    processing_message="Change detection complete!"
                )

            except Exception as e:
                return render_template("image_change.html",error_message=str(e))

    return render_template("image_change.html")



@app.route("/night-detection", methods=["GET", "POST"])
def night_detection():
    processed_image = None
    processed_video = None
    excel_file = None
    processing_message = ""  # Default to empty

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            try:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    processed_image, excel_file = night_module.process_image(file_path)
                elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):
                    processed_video, excel_file = night_module.process_video(file_path)

                processing_message = "Processing complete!"

            except Exception as e:
                return render_template("night_detection.html", error_message=str(e))

    return render_template(
        "night_detection.html",
        processed_image=processed_image,
        processed_video=processed_video,
        excel_file=excel_file,
        processing_message=processing_message,
    )

@app.route("/weapon-detection", methods=["GET", "POST"])
def weapon_detection():
    processed_image = None
    processed_video = None
    excel_file = None
    processing_message = ""

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            try:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    processed_image, excel_file = weapon_module.process_image(file_path)
                elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):
                    processed_video, excel_file = weapon_module.process_video(file_path)

                processing_message = "Weapon detection completed!"

            except Exception as e:
                return render_template("weapons.html", error_message=str(e))

    return render_template(
        "weapons.html",
        processed_image=processed_image,  # just filename like "processed_weapon_*.jpg"
        processed_video=processed_video,
        excel_file=os.path.basename(excel_file) if excel_file else None,
        processing_message=processing_message
    )


@app.route('/results/<path:filename>')
def serve_video(filename):
    results_folder = os.path.join(app.root_path, 'static/results')
    file_path = os.path.join(results_folder, filename)

    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found - {file_path}")
        abort(404)

    print(f"✅ Serving File: {file_path}")  # Debugging

    # Explicitly set content type
    return send_from_directory(results_folder, filename, mimetype="video/mp4")




# ✅ Universal Excel file download route
@app.route("/download_excel/<filename>")
def download_excel(filename):
    possible_dirs = [
        os.getcwd(),  # Root directory (if stored here)
        os.path.join("static", "results"),  # Results folder
        os.path.join("static", "uploads"),  # Uploads folder (if needed)
    ]

    for directory in possible_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)

    return abort(404, "File not found!")  # Return 404 if file isn't found


# ✅ Universal Video file download route
@app.route("/download_video/<filename>")
def download_video(filename):
    possible_dirs = [
        os.getcwd(),
        os.path.join("static", "results"),
        os.path.join("static", "uploads"),
    ]

    for directory in possible_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)

    return abort(404, "File not found!")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port=port
    )
