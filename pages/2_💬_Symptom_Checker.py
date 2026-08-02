import streamlit as st
from backend import load_all_models, predict_nlp

st.set_page_config(
    page_title="Symptom Checker — Medbuddy",
    page_icon="💬",
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

.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #34d399, #60a5fa);
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

/* Text area */
textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
}
textarea:focus {
    border-color: rgba(52,211,153,0.6) !important;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.1) !important;
}
textarea::placeholder { color: #475569 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #059669, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

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
    color: #34d399;
    margin-bottom: 0.25rem;
}
.result-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f0fdf4;
}

/* Example chips */
.example-chip {
    display: inline-block;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 99px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    color: #34d399;
    margin: 0.2rem;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ---- LOAD MODELS (CACHED) ----
@st.cache_resource(show_spinner="Loading AI models…")
def get_models():
    return load_all_models()

models_dict = get_models()
nlp_ready = models_dict.get("nlp_model") is not None and models_dict.get("nlp_label_encoder") is not None


# ---- HEADER ----
st.markdown('<div class="page-title">💬 Symptom Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Describe your symptoms in plain English and receive an AI-powered condition prediction.</div>', unsafe_allow_html=True)

if not nlp_ready:
    st.error("❌ The NLP model or label encoder could not be loaded. Please check the `models/` directory.")
    st.stop()

# ---- EXAMPLE SYMPTOMS ----
st.markdown("**Try an example:**")
examples = [
    "fever, cough, difficulty breathing",
    "chest pain, shortness of breath, sweating",
    "skin rash, itching, redness",
    "headache, stiff neck, sensitivity to light",
    "joint pain, swelling, morning stiffness",
]

# Use session state to pre-fill textarea from examples
if "symptom_input" not in st.session_state:
    st.session_state["symptom_input"] = ""

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["symptom_input"] = ex

st.markdown("<br>", unsafe_allow_html=True)

# ---- INPUT ----
col_input, col_right = st.columns([3, 2], gap="large")

with col_input:
    symptoms_text = st.text_area(
        "Describe your symptoms",
        value=st.session_state["symptom_input"],
        placeholder="e.g. I have been experiencing persistent headache, fever above 38°C, and a stiff neck for the past two days…",
        height=160,
        label_visibility="collapsed",
    )
    st.caption("💡 Be as descriptive as possible — include duration, severity, and any other relevant details.")
    st.markdown("<br>", unsafe_allow_html=True)
    analyse_btn = st.button("🔍 Check Symptoms", disabled=not symptoms_text.strip())

with col_right:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
         border-radius:14px; padding:1.25rem 1.5rem; height:100%;">
        <div style="font-size:0.85rem; font-weight:600; color:#60a5fa; margin-bottom:0.75rem;">
            💡 Tips for better results
        </div>
        <ul style="font-size:0.82rem; color:#94a3b8; margin:0; padding-left:1.2rem; line-height:1.8;">
            <li>List all symptoms, not just the main one</li>
            <li>Include duration (e.g. "for 3 days")</li>
            <li>Mention severity (mild, moderate, severe)</li>
            <li>Add any relevant medical history</li>
            <li>Separate symptoms with commas</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---- INFERENCE ----
if analyse_btn and symptoms_text.strip():
    with st.spinner("Analysing symptoms…"):
        try:
            label, confidence, tips = predict_nlp(symptoms_text.strip(), models_dict)
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Condition</div>
            <div class="result-value">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Model Confidence</div>
            <div class="result-value">{confidence * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence bar
    conf_color = "#22c55e" if confidence >= 0.75 else ("#f59e0b" if confidence >= 0.5 else "#ef4444")
    st.markdown(f"""
    <div style="margin-bottom:0.5rem; font-size:0.8rem; color:#94a3b8; font-weight:500;">CONFIDENCE METER</div>
    <div style="background:rgba(255,255,255,0.08); border-radius:99px; height:12px; overflow:hidden;">
        <div style="width:{confidence*100:.1f}%; background:{conf_color}; height:100%; border-radius:99px;
                    transition:width 0.5s; box-shadow:0 0 8px {conf_color}88;"></div>
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-top:0.3rem;">
        {confidence*100:.1f}% — {'High confidence' if confidence>=0.75 else 'Medium confidence' if confidence>=0.5 else 'Low confidence'}
    </div>
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
