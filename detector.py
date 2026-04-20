from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageChops, ImageFilter, ImageStat


MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "deepfake_model.onnx"
TRAINING_METADATA_PATH = MODEL_DIR / "training_metrics.json"


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    image_size: tuple[int, int] = (224, 224)
    fake_threshold: float = 0.5
    uncertain_margin: float = 0.12


CONFIG = DetectorConfig()


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _load_training_metadata() -> dict[str, Any]:
    if not TRAINING_METADATA_PATH.exists():
        return {}

    try:
        return json.loads(TRAINING_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _active_fake_threshold() -> float:
    metadata = _load_training_metadata()
    value = metadata.get("fake_threshold", CONFIG.fake_threshold)
    try:
        return float(value)
    except (TypeError, ValueError):
        return CONFIG.fake_threshold


def _active_uncertain_margin() -> float:
    metadata = _load_training_metadata()
    value = metadata.get("uncertain_margin", CONFIG.uncertain_margin)
    try:
        return float(value)
    except (TypeError, ValueError):
        return CONFIG.uncertain_margin


def _open_image(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception as exc:  # pragma: no cover - Pillow raises different exceptions per format
        raise ValueError("The uploaded file is not a valid image.") from exc


def _detect_primary_face(image: Image.Image) -> tuple[Image.Image, str]:
    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return image, "Face detector could not be loaded. Full image used."

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return image, "No face detected. Full image used."

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_w = int(w * 0.18)
    pad_h = int(h * 0.18)

    x1 = max(x - pad_w, 0)
    y1 = max(y - pad_h, 0)
    x2 = min(x + w + pad_w, image.width)
    y2 = min(y + h + pad_h, image.height)

    cropped = image.crop((x1, y1, x2, y2))
    if len(faces) == 1:
        return cropped, f"Primary face crop used: {x2 - x1}x{y2 - y1}"

    return (
        cropped,
        f"{len(faces)} faces detected. Largest face crop used: {x2 - x1}x{y2 - y1}",
    )


def _label_from_fake_score(fake_score: float) -> str:
    threshold = _active_fake_threshold()
    margin = _active_uncertain_margin()
    lower_bound = threshold - margin
    upper_bound = threshold + margin

    if lower_bound <= fake_score <= upper_bound:
        return "Uncertain"

    return "Potentially Fake" if fake_score > upper_bound else "Likely Real"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _class_names_from_metadata() -> list[str]:
    metadata = _load_training_metadata()
    class_names = metadata.get("class_names")

    if isinstance(class_names, list) and len(class_names) >= 2:
        return [str(name).lower() for name in class_names[:2]]

    # Training uses torchvision ImageFolder, which sorts class folders alphabetically.
    return ["fake", "real"]


def _normalize_scores(raw_output: np.ndarray) -> tuple[float, float]:
    flat = np.asarray(raw_output, dtype=np.float32).reshape(-1)

    if flat.size == 1:
        fake_score = float(_clamp(flat[0]))
        return 1.0 - fake_score, fake_score

    if flat.size >= 2:
        scores = _softmax(flat[:2])
        class_names = _class_names_from_metadata()
        score_map = {name: float(score) for name, score in zip(class_names, scores)}
        fake_score = score_map.get("fake")
        real_score = score_map.get("real")

        if fake_score is None or real_score is None:
            raise ValueError(
                f"Expected class names to include 'fake' and 'real', got {class_names}"
            )

        return real_score, fake_score

    raise ValueError("The trained model returned an empty prediction.")


def _prepare_model_input(image: Image.Image) -> np.ndarray:
    resized = image.resize(CONFIG.image_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (array - 0.5) / 0.5
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _load_session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _run_model_inference(image: Image.Image, model_path: Path) -> dict[str, Any]:
    session = _load_session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    cropped_image, crop_note = _detect_primary_face(image)
    model_input = _prepare_model_input(cropped_image)
    raw_output = session.run([output_name], {input_name: model_input})[0]
    real_score, fake_score = _normalize_scores(raw_output)

    confidence = round(max(real_score, fake_score) * 100, 1)
    score = round(fake_score, 2)
    label = _label_from_fake_score(fake_score)

    return {
        "label": label,
        "score": score,
        "confidence": confidence,
        "real_probability": round(real_score * 100, 1),
        "fake_probability": round(fake_score * 100, 1),
        "details": [
            f"Model path: {model_path.name}",
            f"Input shape: 1x3x{CONFIG.image_size[0]}x{CONFIG.image_size[1]}",
            f"Real probability: {real_score:.2f}",
            f"Fake probability: {fake_score:.2f}",
            f"Decision band: uncertain between {_active_fake_threshold() - _active_uncertain_margin():.2f} and {_active_fake_threshold() + _active_uncertain_margin():.2f}",
            crop_note,
            "Inference source: ONNX trained deepfake model",
        ],
        "source": "trained_model",
        "model_status": f"Loaded trained model: {model_path.name}",
    }


def _run_heuristic_analysis(image: Image.Image) -> dict[str, Any]:
    cropped_image, crop_note = _detect_primary_face(image)
    grayscale = cropped_image.convert("L")
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(grayscale, grayscale.filter(ImageFilter.GaussianBlur(radius=2)))

    brightness = ImageStat.Stat(grayscale).mean[0] / 255.0
    contrast = ImageStat.Stat(grayscale).stddev[0] / 128.0
    edge_density = ImageStat.Stat(edges).mean[0] / 255.0
    noise_level = ImageStat.Stat(diff).mean[0] / 255.0

    saturation_channels = ImageStat.Stat(cropped_image).mean
    channel_gap = (max(saturation_channels) - min(saturation_channels)) / 255.0

    suspicious_score = 0.0
    suspicious_score += _clamp((0.09 - edge_density) / 0.09) * 0.35
    suspicious_score += _clamp((0.05 - noise_level) / 0.05) * 0.25
    suspicious_score += _clamp((contrast - 0.55) / 0.45) * 0.2
    suspicious_score += _clamp((channel_gap - 0.18) / 0.3) * 0.2

    suspicious_score = round(_clamp(suspicious_score), 2)
    confidence = round(55 + suspicious_score * 40, 1)
    label = _label_from_fake_score(suspicious_score)

    return {
        "label": label,
        "score": suspicious_score,
        "confidence": confidence,
        "real_probability": round((1 - suspicious_score) * 100, 1),
        "fake_probability": round(suspicious_score * 100, 1),
        "details": [
            f"Brightness level: {brightness:.2f}",
            f"Contrast level: {contrast:.2f}",
            f"Edge density: {edge_density:.2f}",
            f"Texture noise: {noise_level:.2f}",
            f"Color channel gap: {channel_gap:.2f}",
            f"Decision band: uncertain between {_active_fake_threshold() - _active_uncertain_margin():.2f} and {_active_fake_threshold() + _active_uncertain_margin():.2f}",
            crop_note,
            "Inference source: heuristic image-forensics fallback",
        ],
        "source": "heuristic",
        "model_status": "No trained model found. Using heuristic fallback.",
    }


def get_model_status() -> dict[str, Any]:
    metadata = _load_training_metadata()

    if CONFIG.model_path.exists():
        return {
            "mode": "trained_model",
            "message": f"Trained model ready: {CONFIG.model_path.name}",
            "summary": metadata.get("summary", "Using ONNX trained deepfake classifier."),
            "best_val_accuracy": metadata.get("best_val_accuracy"),
            "best_epoch": metadata.get("best_epoch"),
            "epochs": metadata.get("epochs"),
            "model_name": CONFIG.model_path.name,
            "fake_threshold": _active_fake_threshold(),
            "uncertain_margin": _active_uncertain_margin(),
        }

    return {
        "mode": "heuristic",
        "message": "Add models/deepfake_model.onnx to enable trained-model inference.",
        "summary": "Heuristic fallback is active until a trained ONNX model is available.",
        "best_val_accuracy": metadata.get("best_val_accuracy"),
        "best_epoch": metadata.get("best_epoch"),
        "epochs": metadata.get("epochs"),
        "model_name": None,
        "fake_threshold": _active_fake_threshold(),
        "uncertain_margin": _active_uncertain_margin(),
    }


def analyze_image(image_path: Path) -> dict[str, Any]:
    image = _open_image(image_path)

    if CONFIG.model_path.exists():
        try:
            return _run_model_inference(image, CONFIG.model_path)
        except Exception as exc:
            fallback = _run_heuristic_analysis(image)
            fallback["details"].append(f"Model load failed, fallback used: {exc}")
            fallback["model_status"] = "Trained model detected but failed to run. Using heuristic fallback."
            fallback["source"] = "fallback_after_model_error"
            return fallback

    return _run_heuristic_analysis(image)
