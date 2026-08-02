import streamlit as st
from backend import (
    load_all_models,
    predict_eye,
    predict_keras_vision,
    DENTAL_CLASSES,
    SKIN_CLASSES,
)

st.set_page_config(
    page_title="Image Analysis — Medbuddy",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- GLOBAL CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

h1, h2, h3, h4, p, label, div { color: #e2e8f0; }

/* Upload zone */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(167,139,250,0.4);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.8);
}

/* Result card */
.result-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.75rem;
    backdrop-filter: blur(12px);
    margin-top: 1rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a78bfa;
    margin-bottom: 0.25rem;
}
.result-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f0fdf4;
    margin-bottom: 0;
}

/* Tips expander */
.tips-section h4 {
    color: #60a5fa;
    font-weight: 600;
    margin-top: 1rem;
}
.tips-section ul {
    margin: 0.25rem 0 0.75rem 1.25rem;
    color: #cbd5e1;
    line-height: 1.7;
}

/* Selectbox override */
[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---- LOAD MODELS (CACHED) ----
@st.cache_resource(show_spinner="Loading AI models… this may take a moment on first run.")
def get_models():
    return load_all_models()

models_dict = get_models()


# ---- HEADER ----
st.markdown('<div class="page-title">🔬 Medical Image Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Upload an image and select the analysis type to receive an AI-powered diagnosis.</div>', unsafe_allow_html=True)

# ---- CONTROLS ----
col_sel, col_gap = st.columns([1, 2])
with col_sel:
    analysis_type = st.selectbox(
        "Analysis Type",
        options=["Eye Disease", "Dental / Oral Disease", "Skin / Dermatological Disease"],
        index=0,
        help="Choose which AI model to use for analysis.",
    )

model_key_map = {
    "Eye Disease": "eye",
    "Dental / Oral Disease": "dental",
    "Skin / Dermatological Disease": "skin",
}
model_key = model_key_map[analysis_type]

# Model availability check
model_available = {
    "eye": models_dict.get("eye_model") is not None,
    "dental": models_dict.get("dental_model") is not None,
    "skin": models_dict.get("skin_model") is not None,
}

if not model_available[model_key]:
    st.warning(f"⚠️ The **{analysis_type}** model could not be loaded. Check that the model file exists in the `models/` directory.", icon="⚠️")

st.markdown("<br>", unsafe_allow_html=True)

# ---- FILE UPLOAD ----
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, PNG, BMP, WEBP",
        label_visibility="collapsed",
    )
    st.caption("📎 Drag & drop or click to browse — JPG, PNG, BMP, WEBP")

    if uploaded_file:
        run_btn = st.button("🚀 Run Analysis", use_container_width=True, disabled=not model_available[model_key])

with col_preview:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# ---- INFERENCE ----
if uploaded_file and model_available[model_key]:
    if "run_btn" in dir() and run_btn:
        image_bytes = uploaded_file.read()

        with st.spinner("Analysing image…"):
            try:
                if model_key == "eye":
                    label, confidence, tips = predict_eye(image_bytes, models_dict)
                elif model_key == "dental":
                    label, confidence, tips = predict_keras_vision(
                        image_bytes, models_dict["dental_model"], DENTAL_CLASSES, models_dict, "dental"
                    )
                else:  # skin
                    label, confidence, tips = predict_keras_vision(
                        image_bytes, models_dict["skin_model"], SKIN_CLASSES, models_dict, "skin"
                    )
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                st.stop()

        # ---- RESULTS ----
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Condition</div>
                <div class="result-value">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Model Confidence</div>
                <div class="result-value">{confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        conf_color = "#22c55e" if confidence >= 0.75 else ("#f59e0b" if confidence >= 0.5 else "#ef4444")
        st.markdown(f"""
        <div style="margin-bottom:0.5rem; font-size:0.8rem; color:#94a3b8; font-weight:500;">CONFIDENCE METER</div>
        <div style="background:rgba(255,255,255,0.08); border-radius:99px; height:12px; overflow:hidden;">
            <div style="width:{confidence*100:.1f}%; background:{conf_color}; height:100%; border-radius:99px;
                        transition:width 0.5s ease; box-shadow:0 0 8px {conf_color}88;"></div>
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.3rem;">{confidence*100:.1f}% — {'High confidence' if confidence>=0.75 else 'Medium confidence' if confidence>=0.5 else 'Low confidence'}</div>
        """, unsafe_allow_html=True)

        # ---- TIPS ----
        if tips:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 Clinical Information & Tips", expanded=True):
                severity = tips.get("severity_estimate", "—")
                st.markdown(f"**Severity Estimate:** `{severity}`")

                if causes := tips.get("possible_causes"):
                    st.markdown("#### 🔎 Possible Causes")
                    for c in causes:
                        st.markdown(f"- {c}")

                if symptoms := tips.get("common_symptoms"):
                    st.markdown("#### 🤒 Common Symptoms")
                    for s in symptoms:
                        st.markdown(f"- {s}")

                if treatments := tips.get("treatment_suggestions"):
                    st.markdown("#### 💊 Treatment Suggestions")
                    for t in treatments:
                        st.markdown(f"- {t}")

                if prevention := tips.get("prevention_tips"):
                    st.markdown("#### 🛡️ Prevention Tips")
                    for p in prevention:
                        st.markdown(f"- {p}")

                if when := tips.get("when_to_see_a_doctor"):
                    st.info(f"🏥 **When to see a doctor:** {when}")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("ℹ️ No additional clinical tips available for this condition.")

# ---- DISCLAIMER ----
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.2);
     border-radius:12px; padding:0.8rem 1.25rem; text-align:center;">
    <span style="font-size:0.78rem; color:#fbbf24;">
        ⚠️ <strong>Disclaimer:</strong> Results are AI-generated for educational purposes only.
        Always consult a qualified healthcare professional for medical advice.
    </span>
</div>
""", unsafe_allow_html=True)
