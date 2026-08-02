import io
import os

import warnings

# Suppress TensorFlow logs and force CPU usage to save memory
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress noisy warnings from TF, sklearn, and torch
warnings.filterwarnings("ignore")

import json
import tempfile
import zipfile
from typing import Any, Dict, Optional

import joblib
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms

import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Medbuddy API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 1. MODEL LOADING ----

# Helper function to preprocess images inside Lambda layers if used during training
def custom_preprocess_input(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)


# Compat wrappers: accept legacy 'quantization_config' kwarg from older Keras models
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


# Custom Objects Mapping to solve Keras deserialization issues
CUSTOM_OBJECTS = {
    "preprocess_input": custom_preprocess_input,
    "Dense": CompatDense,
    "Embedding": CompatEmbedding,
}

BASE_MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

# A. Eye Disease Model (PyTorch)
EYE_CLASSES = ["Cataract", "Normal_Eye", "Pterygium"]

# Dental disease classes (case-sensitive sorted, matching training dataset folder order)
DENTAL_CLASSES = [
    "Calculus",             # 0
    "Hypodontia",           # 1
    "Mouth ulcer",          # 2
    "Normal",               # 3
    "caries",               # 4
    "tooth discoloration",  # 5
]

device = torch.device("cpu")
eye_model = models.resnet18(weights=None)
num_ftrs = eye_model.fc.in_features
eye_model.fc = nn.Linear(num_ftrs, len(EYE_CLASSES))

eye_model_path = os.path.join(BASE_MODEL_DIR, "eye_disease_model.pth")
try:
    if os.path.exists(eye_model_path):
        eye_model.load_state_dict(
            torch.load(eye_model_path, map_location=device)
        )
        eye_model.eval()
        print("Eye disease model loaded successfully.")
    else:
        print(f"Eye model file not found at: {eye_model_path}")
        eye_model = None
except Exception as e:
    print(f"Failed to load eye disease model: {e}")
    eye_model = None

eye_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _patch_keras_config(obj):
    """Recursively patch Keras config JSON to fix cross-version incompatibilities."""
    if isinstance(obj, list):
        return [_patch_keras_config(item) for item in obj]
    if not isinstance(obj, dict):
        return obj

    # Fix DTypePolicy: convert complex dtype dict to a simple string
    if obj.get("class_name") in ("DTypePolicy", "keras.DTypePolicy"):
        return obj.get("config", {}).get("name", "float32")

    patched = {k: _patch_keras_config(v) for k, v in obj.items()}

    cfg = patched.get("config", {})
    class_name = patched.get("class_name", "")

    # Fix InputLayer: 'batch_shape' -> 'shape' (strip batch dim)
    if class_name == "InputLayer":
        if "batch_shape" in cfg and "shape" not in cfg:
            batch_shape = cfg.pop("batch_shape")
            cfg["shape"] = batch_shape[1:] if batch_shape else []

    # Remove 'quantization_config' — not supported in current Keras 3
    # (was stored as None in older Keras versions for Dense, Embedding, etc.)
    cfg.pop("quantization_config", None)

    # Remove 'data_format' from augmentation layers that didn't have it originally
    if class_name in ("RandomFlip", "RandomRotation", "RandomZoom"):
        cfg.pop("data_format", None)

    return patched


# Helper function to load Keras models safely with fallback options
def load_keras_model_safely(model_path):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None

    # --- Attempt 1: Direct load ---
    try:
        return tf.keras.models.load_model(
            model_path, custom_objects=CUSTOM_OBJECTS, safe_mode=False
        )
    except Exception:
        pass

    # --- Attempt 2: Patch the config JSON inside the .keras zip ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # .keras files are zip archives
            with zipfile.ZipFile(model_path, "r") as zf:
                zf.extractall(tmpdir)

            config_file = os.path.join(tmpdir, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                patched = _patch_keras_config(config)
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(patched, f)

            # Repack as a new .keras zip
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


# B. Keras Vision Models (Dental & Skin)
dental_model = None
dental_h5_path = os.path.join(BASE_MODEL_DIR, "oral_disease_model2.h5")
try:
    dental_model = tf.keras.models.load_model(
        dental_h5_path, custom_objects=CUSTOM_OBJECTS, safe_mode=False
    )
    print("Dental model loaded successfully.")
except Exception as e:
    print(f"Failed to load dental model: {e}")

skin_model = None
skin_h5_path = os.path.join(BASE_MODEL_DIR, "skin_disease_model2.h5")
try:
    skin_model = tf.keras.models.load_model(
        skin_h5_path, custom_objects=CUSTOM_OBJECTS, safe_mode=False
    )
    print("Skin model loaded successfully.")
except Exception as e:
    print(f"Failed to load skin model: {e}")

# C. NLP Model & Label Encoder
nlp_model = load_keras_model_safely(
    os.path.join(BASE_MODEL_DIR, "nlp_symptom_disease.keras")
)
nlp_encoder_path = os.path.join(BASE_MODEL_DIR, "label_encoder.joblib")

try:
    if os.path.exists(nlp_encoder_path):
        nlp_label_encoder = joblib.load(nlp_encoder_path)
        print("NLP Label Encoder loaded successfully.")
    else:
        nlp_label_encoder = None
except Exception as e:
    print(f"Failed to load NLP Label Encoder: {e}")
    nlp_label_encoder = None

if nlp_model and nlp_label_encoder:
    print("NLP Pipeline ready.")


# D. Disease Tips Data Loader
TIPS_DIR = os.path.join(os.path.dirname(__file__), "tips")
EYE_TIPS = []
DENTAL_TIPS = []
SKIN_TIPS = []

try:
    eye_tips_path = os.path.join(TIPS_DIR, "eye_diseases_analysis.json")
    if os.path.exists(eye_tips_path):
        with open(eye_tips_path, "r", encoding="utf-8") as f:
            EYE_TIPS = json.load(f)
            
    dental_tips_path = os.path.join(TIPS_DIR, "dental_diseases_analysis_no_confidence.json")
    if os.path.exists(dental_tips_path):
        with open(dental_tips_path, "r", encoding="utf-8") as f:
            DENTAL_TIPS = json.load(f)
            
    skin_tips_path = os.path.join(TIPS_DIR, "dermatology_diseases_analysis.json")
    if os.path.exists(skin_tips_path):
        with open(skin_tips_path, "r", encoding="utf-8") as f:
            SKIN_TIPS = json.load(f)
    print("Disease tips database loaded successfully.")
except Exception as e:
    print(f"Failed to load disease tips: {e}")


def get_disease_tips(model_type: str, prediction: str) -> Optional[Dict[str, Any]]:
    if model_type == "eye":
        tips_list = EYE_TIPS
    elif model_type == "dental":
        tips_list = DENTAL_TIPS
    elif model_type == "skin":
        tips_list = SKIN_TIPS
    else:
        tips_list = EYE_TIPS + DENTAL_TIPS + SKIN_TIPS

    clean_pred = prediction.lower().replace("normal_eye", "normal").replace("_", "").replace(" ", "")
    for t in tips_list:
        clean_name = t.get("disease_name", "").lower().replace("_", "").replace(" ", "")
        if clean_pred in clean_name or clean_name in clean_pred:
            return t
    return None


# ---- 2. PYDANTIC SCHEMAS ----
class PredictionResponse(BaseModel):
    model_used: str
    prediction: str
    confidence: float
    tips: Optional[Dict[str, Any]] = None


class NLPRequest(BaseModel):
    symptoms: str


# ---- 3. INFERENCE HELPER FUNCTIONS ----
def predict_eye(image_bytes: bytes) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = eye_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = eye_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    return EYE_CLASSES[predicted_idx.item()], float(confidence.item())


def predict_keras_vision(
    image_bytes: bytes, model, class_names, target_size=(224, 224)
) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Shape: (1, H, W, 3)

    predictions = model.predict(img_array, verbose=0)

    # Check if final layer is Softmax or Raw Logits
    if np.max(predictions[0]) > 1.0 or np.min(predictions[0]) < 0.0:
        probs = tf.nn.softmax(predictions[0]).numpy()
    else:
        probs = predictions[0]

    class_idx = np.argmax(probs)
    confidence = float(np.max(probs))

    label = (
        class_names[class_idx]
        if class_idx < len(class_names)
        else f"Condition Class {class_idx}"
    )
    return label, confidence


# ---- 4. FASTAPI ENDPOINTS ----
@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...), model_type: str = Form(...)
):
    if model_type not in ["eye", "dental", "skin"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid model_type. Choose from: eye, dental, skin",
        )

    image_bytes = await file.read()

    try:
        if model_type == "eye":
            if eye_model is None:
                raise HTTPException(
                    status_code=500, detail="Eye model is not available."
                )
            prediction, confidence = predict_eye(image_bytes)

        elif model_type == "dental":
            if dental_model is None:
                raise HTTPException(
                    status_code=500, detail="Dental model is not available."
                )
            prediction, confidence = predict_keras_vision(
                image_bytes, dental_model, DENTAL_CLASSES, target_size=(224, 224)
            )

        elif model_type == "skin":
            if skin_model is None:
                raise HTTPException(
                    status_code=500, detail="Skin model is not available."
                )
            # Class names sorted alphabetically (matches ImageDataGenerator class_indices order)
            skin_classes = [
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
            prediction, confidence = predict_keras_vision(
                image_bytes, skin_model, skin_classes, target_size=(224, 224)
            )

        return PredictionResponse(
            model_used=model_type,
            prediction=prediction,
            confidence=round(confidence, 4),
            tips=get_disease_tips(model_type, prediction),
        )
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/nlp", response_model=PredictionResponse)
async def predict_nlp(request: NLPRequest):
    if nlp_model is None or nlp_label_encoder is None:
        raise HTTPException(
            status_code=500, detail="NLP Model or Label Encoder not available."
        )

    if not request.symptoms.strip():
        raise HTTPException(
            status_code=400, detail="Symptom query cannot be empty."
        )

    try:
        # Use tf.constant with string dtype — np.array creates Unicode dtype which Keras rejects
        input_data = tf.constant([request.symptoms], dtype=tf.string)
        predictions = nlp_model.predict(input_data, verbose=0)

        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        predicted_disease = nlp_label_encoder.inverse_transform([class_idx])[0]

        return PredictionResponse(
            model_used="nlp",
            prediction=str(predicted_disease),
            confidence=round(confidence, 4),
            tips=get_disease_tips("nlp", str(predicted_disease)),
        )
    except Exception as e:
        print(f"NLP Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- 5. STATIC FILES & MOUNTING ----
# Serve web application from web_app/static
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_app", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount(
    "/", StaticFiles(directory=STATIC_DIR, html=True), name="static"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)