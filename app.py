from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from detector import analyze_image, get_model_status


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "UPLOADS"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__, template_folder="TEMPLATES")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_saved_filename(filename: str) -> str:
    safe_name = secure_filename(filename)
    stem = Path(safe_name).stem or "upload"
    extension = Path(safe_name).suffix.lower()
    return f"{stem}-{uuid4().hex[:10]}{extension}"


@app.route("/")
def home():
    return render_template("index.html", detector_status=get_model_status())


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    file_storage = request.files.get("file")

    if not file_storage or file_storage.filename == "":
        return render_template(
            "index.html",
            error="Please choose an image first.",
            detector_status=get_model_status(),
        )

    if not allowed_file(file_storage.filename):
        return render_template(
            "index.html",
            error="Only PNG, JPG, JPEG, and WEBP images are supported.",
            detector_status=get_model_status(),
        )

    original_filename = secure_filename(file_storage.filename) or file_storage.filename
    saved_filename = build_saved_filename(file_storage.filename)
    filepath = UPLOAD_DIR / saved_filename
    file_storage.save(filepath)

    try:
        analysis = analyze_image(filepath)
    except ValueError as exc:
        if filepath.exists():
            filepath.unlink()
        return render_template(
            "index.html",
            error=str(exc),
            detector_status=get_model_status(),
        )

    detector_status = get_model_status()
    detector_status["mode"] = analysis["source"]
    detector_status["message"] = analysis["model_status"]

    return render_template(
        "index.html",
        result=analysis["label"],
        confidence=analysis["confidence"],
        score=analysis["score"],
        real_probability=analysis["real_probability"],
        fake_probability=analysis["fake_probability"],
        details=analysis["details"],
        filename=original_filename,
        stored_filename=saved_filename,
        uploaded_image_url=url_for("uploaded_file", filename=saved_filename),
        detector_status=detector_status,
    )


if __name__ == "__main__":
    app.run(debug=True)
