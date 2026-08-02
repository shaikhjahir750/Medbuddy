import streamlit as st

st.set_page_config(
    page_title="Medbuddy — AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- GLOBAL CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main content text */
h1, h2, h3, h4, h5, p, label, div {
    color: #e2e8f0;
}

/* Hero section */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.25rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
}

/* Feature cards */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.25rem;
    margin: 2rem 0;
}
.feature-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.75rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
    backdrop-filter: blur(10px);
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(167,139,250,0.5);
}
.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.feature-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.feature-desc {
    font-size: 0.875rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Disclaimer */
.disclaimer {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 2rem;
}
.disclaimer p {
    font-size: 0.8rem;
    color: #fbbf24;
    margin: 0;
    text-align: center;
}

/* Stat badges */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.stat-badge {
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ---- HERO ----
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🩺 Medbuddy</div>
    <div class="hero-subtitle">
        AI-powered medical image analysis and symptom diagnosis.<br>
        Fast, accurate, and built to assist — not replace — your doctor.
    </div>
</div>
""", unsafe_allow_html=True)

# ---- STATS ----
st.markdown("""
<div class="stat-row">
    <div class="stat-badge">
        <div class="stat-number">4</div>
        <div class="stat-label">AI Models</div>
    </div>
    <div class="stat-badge">
        <div class="stat-number">22+</div>
        <div class="stat-label">Conditions Detected</div>
    </div>
    <div class="stat-badge">
        <div class="stat-number">3</div>
        <div class="stat-label">Imaging Specialties</div>
    </div>
    <div class="stat-badge">
        <div class="stat-number">CPU</div>
        <div class="stat-label">Optimised Inference</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---- FEATURE CARDS ----
st.markdown("""
<div class="card-grid">
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Image Analysis</div>
        <div class="feature-desc">Upload a medical image and let AI detect Eye, Dental, or Skin conditions with confidence scores.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">Symptom Checker</div>
        <div class="feature-desc">Describe your symptoms in plain text and receive a predicted diagnosis powered by NLP.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📋</div>
        <div class="feature-title">Clinical Tips</div>
        <div class="feature-desc">Every result comes with causes, symptoms, treatment suggestions, and when to see a doctor.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- NAVIGATION HINT ----
st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Use the **sidebar** to navigate to **Image Analysis** or **Symptom Checker**.")

# ---- DISCLAIMER ----
st.markdown("""
<div class="disclaimer">
    <p>⚠️ <strong>Medical Disclaimer:</strong> Medbuddy is an AI research tool for educational purposes only.
    It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.</p>
</div>
""", unsafe_allow_html=True)
