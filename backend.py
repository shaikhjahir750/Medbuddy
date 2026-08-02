import io
import os
import json
import tempfile
import warnings
import zipfile
from typing import Any, Dict, Optional

# Suppress TensorFlow logs and force CPU usage to save memory
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import tensorflow as tf


# ---- MODEL & CLASS DEFINITIONS ----

BASE_MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
TIPS_DIR = os.path.join(os.path.dirname(__file__), "tips")

EYE_CLASSES = ["Cataract", "Normal_Eye", "Pterygium"]

DENTAL_CLASSES = [
    "Calculus",
    "Hypodontia",
    "Mouth ulcer",
    "Normal",
    "caries",
    "tooth discoloration",
]

SKIN_CLASSES = [
    "Eczema",
    "Herpes",
    "Impetigo",
    "Psoriasis",
    "Rashes",
    "Ringworm",
    "Rosacea",
    "Scabies",
    "Vitiligo",
    "Acne",
    "Cellulitis",
    "Dark Spots / Hyperpigmentation",
    "Normal",
    "Skin Cancer",
    "Urticaria / Hives",
    "Wrinkle",
]


# ---- KERAS COMPAT HELPERS ----

def custom_preprocess_input(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)


class CompatDense(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


class CompatEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


CUSTOM_OBJECTS = {
    "preprocess_input": custom_preprocess_input,
    "Dense": CompatDense,
    "Embedding": CompatEmbedding,
}


def _patch_keras_config(obj):
    """Recursively patch Keras config JSON to fix cross-version incompatibilities."""
    if isinstance(obj, list):
        return [_patch_keras_config(item) for item in obj]
    if not isinstance(obj, dict):
        return obj

    if obj.get("class_name") in ("DTypePolicy", "keras.DTypePolicy"):
        return obj.get("config", {}).get("name", "float32")

    patched = {k: _patch_keras_config(v) for k, v in obj.items()}
    cfg = patched.get("config", {})
    class_name = patched.get("class_name", "")

    if class_name == "InputLayer":
        if "batch_shape" in cfg and "shape" not in cfg:
            batch_shape = cfg.pop("batch_shape")
            cfg["shape"] = batch_shape[1:] if batch_shape else []

    cfg.pop("quantization_config", None)

    if class_name in ("RandomFlip", "RandomRotation", "RandomZoom"):
        cfg.pop("data_format", None)

    return patched


def load_keras_model_safely(model_path):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None

    try:
        return tf.keras.models.load_model(
            model_path, custom_objects=CUSTOM_OBJECTS, safe_mode=False
        )
    except Exception:
        pass

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, "r") as zf:
                zf.extractall(tmpdir)

            config_file = os.path.join(tmpdir, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                patched = _patch_keras_config(config)
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(patched, f)

            tmp_keras = model_path + ".patched.keras"
            with zipfile.ZipFile(tmp_keras, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        arcname = os.path.relpath(full_path, tmpdir)
                        zf.write(full_path, arcname)

            model = tf.keras.models.load_model(
                tmp_keras, custom_objects=CUSTOM_OBJECTS, safe_mode=False
            )
            os.remove(tmp_keras)
            return model
    except Exception as e:
        print(f"Failed to load Keras model '{model_path}': {e}")
        return None


# ---- MODEL LOADING ----

def load_all_models():
    """Load all ML models. Meant to be called once via st.cache_resource."""
    loaded = {}

    # Eye model (PyTorch)
    device = torch.device("cpu")
    eye_model = models.resnet18(weights=None)
    num_ftrs = eye_model.fc.in_features
    eye_model.fc = nn.Linear(num_ftrs, len(EYE_CLASSES))

    eye_model_path = os.path.join(BASE_MODEL_DIR, "eye_disease_model.pth")
    try:
        if os.path.exists(eye_model_path):
            eye_model.load_state_dict(torch.load(eye_model_path, map_location=device))
            eye_model.eval()
            loaded["eye_model"] = eye_model
        else:
            loaded["eye_model"] = None
    except Exception as e:
        print(f"Failed to load eye model: {e}")
        loaded["eye_model"] = None

    loaded["device"] = device

    # Dental model (Keras h5)
    try:
        loaded["dental_model"] = tf.keras.models.load_model(
            os.path.join(BASE_MODEL_DIR, "oral_disease_model2.h5"),
            custom_objects=CUSTOM_OBJECTS,
            safe_mode=False,
        )
    except Exception as e:
        print(f"Failed to load dental model: {e}")
        loaded["dental_model"] = None

    # Skin model (Keras h5)
    try:
        loaded["skin_model"] = tf.keras.models.load_model(
            os.path.join(BASE_MODEL_DIR, "skin_disease_model2.h5"),
            custom_objects=CUSTOM_OBJECTS,
            safe_mode=False,
        )
    except Exception as e:
        print(f"Failed to load skin model: {e}")
        loaded["skin_model"] = None

    # NLP model + label encoder
    loaded["nlp_model"] = load_keras_model_safely(
        os.path.join(BASE_MODEL_DIR, "nlp_symptom_disease.keras")
    )
    nlp_encoder_path = os.path.join(BASE_MODEL_DIR, "label_encoder.joblib")
    try:
        loaded["nlp_label_encoder"] = (
            joblib.load(nlp_encoder_path) if os.path.exists(nlp_encoder_path) else None
        )
    except Exception as e:
        print(f"Failed to load NLP encoder: {e}")
        loaded["nlp_label_encoder"] = None

    # Tips data
    try:
        def load_json(path):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []

        loaded["eye_tips"] = load_json(os.path.join(TIPS_DIR, "eye_diseases_analysis.json"))
        loaded["dental_tips"] = load_json(os.path.join(TIPS_DIR, "dental_diseases_analysis_no_confidence.json"))
        loaded["skin_tips"] = load_json(os.path.join(TIPS_DIR, "dermatology_diseases_analysis.json"))
    except Exception as e:
        print(f"Failed to load tips: {e}")
        loaded["eye_tips"] = []
        loaded["dental_tips"] = []
        loaded["skin_tips"] = []

    return loaded


# ---- IMAGE TRANSFORM ----

eye_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---- TIPS LOOKUP ----

def get_disease_tips(model_type: str, prediction: str, models_dict: dict) -> Optional[Dict[str, Any]]:
    if model_type == "eye":
        tips_list = models_dict.get("eye_tips", [])
    elif model_type == "dental":
        tips_list = models_dict.get("dental_tips", [])
    elif model_type == "skin":
        tips_list = models_dict.get("skin_tips", [])
    else:
        tips_list = (
            models_dict.get("eye_tips", [])
            + models_dict.get("dental_tips", [])
            + models_dict.get("skin_tips", [])
        )

    clean_pred = prediction.lower().replace("normal_eye", "normal").replace("_", "").replace(" ", "")
    for t in tips_list:
        clean_name = t.get("disease_name", "").lower().replace("_", "").replace(" ", "")
        if clean_pred in clean_name or clean_name in clean_pred:
            return t
    return None


# ---- INFERENCE FUNCTIONS ----

def predict_eye(image_bytes: bytes, models_dict: dict) -> tuple:
    """Returns (label, confidence, tips_dict)."""
    eye_model = models_dict["eye_model"]
    device = models_dict["device"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = eye_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = eye_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, 0)
    label = EYE_CLASSES[predicted_idx.item()]
    tips = get_disease_tips("eye", label, models_dict)
    return label, float(confidence.item()), tips


def predict_keras_vision(image_bytes: bytes, model, class_names: list, models_dict: dict, model_type: str, target_size=(224, 224)) -> tuple:
    """Returns (label, confidence, tips_dict)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)

    if np.max(predictions[0]) > 1.0 or np.min(predictions[0]) < 0.0:
        probs = tf.nn.softmax(predictions[0]).numpy()
    else:
        probs = predictions[0]

    class_idx = np.argmax(probs)
    confidence = float(np.max(probs))
    label = class_names[class_idx] if class_idx < len(class_names) else f"Condition Class {class_idx}"
    tips = get_disease_tips(model_type, label, models_dict)
    return label, confidence, tips


def predict_nlp(symptoms: str, models_dict: dict) -> tuple:
    """Returns (label, confidence, tips_dict)."""
    nlp_model = models_dict["nlp_model"]
    encoder = models_dict["nlp_label_encoder"]
    input_data = tf.constant([symptoms], dtype=tf.string)
    predictions = nlp_model.predict(input_data, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    label = str(encoder.inverse_transform([class_idx])[0])
    tips = get_disease_tips("nlp", label, models_dict)
    return label, confidence, tips
