# Medbuddy - AI-Powered Health Analysis Platform

Medbuddy is a deep learning-based health analysis platform designed to assist users in evaluating health conditions through computer vision and natural language processing (NLP). The platform provides automated image classification for Eye, Dental, and Dermatology conditions, alongside a text-based symptom checker, and delivers actionable clinical tips and recommendations.

---

## Key Features

1. **Eye Disease Classification (PyTorch)**
   - Powered by a fine-tuned **ResNet18** transfer learning model.
   - Classifies images into **Cataract**, **Pterygium**, or **Normal Eye**.

2. **Oral / Dental Disease Classification (Keras)**
   - Deep vision classification model.
   - Identifies conditions including **Calculus**, **Hypodontia**, **Mouth Ulcers**, **Dental Caries**, **Tooth Discoloration**, or **Normal**.

3. **Skin / Dermatology Disease Classification (Keras)**
   - Classifies 16 different skin conditions (e.g., **Acne**, **Eczema**, **Herpes**, **Impetigo**, **Psoriasis**, **Rashes**, **Ringworm**, **Rosacea**, **Scabies**, **Vitiligo**, **Cellulitis**, **Dark Spots / Hyperpigmentation**, **Skin Cancer**, **Urticaria / Hives**, **Wrinkle**, or **Normal**).

4. **Symptom NLP Analyzer (TensorFlow / Scikit-Learn)**
   - NLP model with Scikit-learn Label Encoding (`label_encoder.joblib`).
   - Processes natural language symptom descriptions and predicts likely medical conditions.

5. **Integrated Medical Advice & Tips Database**
   - Correlates model predictions with structured JSON databases (`tips/`) to provide:
     - Disease Descriptions
     - Recommended Precautions & Next Steps
     - Clinical Advice & Care Tips

6. **Modern Interactive Web Frontend**
   - Built with HTML5, CSS3, and Vanilla JavaScript (`web_app/static/`).
   - Includes dedicated user portals for **Image Analysis** (`image_analysis.html`) and **Symptom Detection** (`symptom_detection.html`).

---

## System Architecture & Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) with [Uvicorn](https://www.uvicorn.org/) server.
- **Deep Learning Libraries**:
  - [PyTorch](https://pytorch.org/) & `torchvision` for the ResNet18 Eye Disease model.
  - [TensorFlow](https://www.tensorflow.org/) / Keras for Dental, Skin, and Symptom NLP models.
- **Machine Learning Utilities**: [Scikit-learn](https://scikit-learn.org/) / `joblib`, [NumPy](https://numpy.org/), [Pillow (PIL)](https://python-pillow.org/).
- **Data Validation**: [Pydantic](https://docs.pydantic.dev/).
- **Frontend**: Responsive Web UI (HTML5, Vanilla JS, CSS3).

---

## Repository Directory Structure

```
Medbuddy/
├── main.py                     # Primary FastAPI server, model loaders, & API endpoints
├── test_api.py                 # API test script using FastAPI TestClient
├── README.md                   # Project documentation
│
├── MODEL_TRAINING/             # Training scripts and Jupyter notebooks
│   ├── eye_disease_model.py    # PyTorch ResNet18 eye model training script
│   ├── oral_disease_model_training.ipynb
│   ├── Skin_disease_mode.ipynb
│   └── text_symptoms_model.ipynb
│
├── models/                     # Pre-trained weights & label encoders
│   ├── eye_disease_model.pth           # PyTorch state dict for eye model
│   ├── oral_disease_model2.h5          # Keras dental model
│   ├── skin_disease_model2.h5          # Keras skin model
│   ├── nlp_symptom_disease.keras       # Keras text classification NLP model
│   └── label_encoder.joblib            # Scikit-learn label encoder for NLP
│
├── tips/                       # JSON databases containing disease tips & recommendations
│   ├── eye_diseases_analysis.json
│   ├── dental_diseases_analysis_no_confidence.json
│   └── dermatology_diseases_analysis.json
│
├── Reports/                    # Model performance reports & metrics
│   └── eye_model_reports/
│       ├── classification_report.txt
│       └── confusion_matrix.png
│
└── web_app/                    # Web frontend assets
    └── static/
        ├── index.html          # Main landing page
        ├── image_analysis.html # Medical image classification portal
        ├── symptom_detection.html # Text symptom checker portal
        ├── script.js           # Frontend API connection & UI logic
        └── style.css           # CSS styling & responsive layout
```

---

## API Endpoints Reference

### 1. Image Classification Endpoint
- **Endpoint**: `POST /predict/image`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Uploaded image file (`.jpg`, `.jpeg`, `.png`)
  - `model_type`: Target domain (`eye`, `dental`, or `skin`)
- **Example Response**:
```json
{
  "model_used": "eye",
  "prediction": "Cataract",
  "confidence": 0.9854,
  "tips": {
    "disease_name": "Cataract",
    "description": "Clouding of the eye's natural lens...",
    "precautions": ["Consult an ophthalmologist", "Wear protective sunglasses"],
    "advice": "Surgical intervention is the standard effective treatment..."
  }
}
```

### 2. Symptom NLP Endpoint
- **Endpoint**: `POST /predict/nlp`
- **Content-Type**: `application/json`
- **Payload**:
```json
{
  "symptoms": "I have severe headache, high fever, and sensitivity to light"
}
```
- **Example Response**:
```json
{
  "model_used": "nlp",
  "prediction": "Migraine",
  "confidence": 0.9120,
  "tips": { ... }
}
```

### 3. Static Frontend Route
- **Endpoint**: `GET /`
- Serves the interactive web interface mounted from `web_app/static/`.

---

## Keras Compatibility Shims

`main.py` includes custom compatibility layers and dynamic model repair routines to ensure seamless deserialization across different Keras versions:
- `CompatDense` & `CompatEmbedding`: Strip unsupported arguments (e.g., legacy `quantization_config`) during layer reconstruction.
- `_patch_keras_config()` & `load_keras_model_safely()`: Automatically unpacks `.keras` archive `config.json` files and patches structural incompatibilities (such as `InputLayer` shape specs) on loading.

---

## Getting Started & Local Setup

### Prerequisites
- Python 3.9+ installed.
- Model weight files placed in the `models/` directory.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shaikhjahir750/Medbuddy.git
   cd Medbuddy
   ```

2. **Install Required Packages**:
   ```bash
   pip install fastapi uvicorn torch torchvision tensorflow pillow numpy pydantic joblib scikit-learn
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```
   *or using uvicorn directly:*
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the Web Interface**:
   Open your browser and navigate to `http://localhost:8000`.

---

## Testing

To verify API endpoints and model pipelines:
```bash
python test_api.py
```
This script sends synthetic test image payloads to the `/predict/image` endpoint to validate status codes and response schemas.

---

## Disclaimer
*Medbuddy is designed for educational and informational assistance only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.*
