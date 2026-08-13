import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Current | Emotion Analysis",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

model_path = "artifacts/BiGRU.h5"
tokenizer_path = "artifacts/tokenizer.pkl"
max_sequence_length = 50

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Each emotion gets its own place in the water column + a colour drawn from it
emotion_meta = {
    "joy":      {"emoji": "☀️", "color": "#ffd23f", "glow": "rgba(255, 210, 63, 0.45)",  "depth": "Sunlit Shallows"},
    "love":     {"emoji": "🐚", "color": "#ff8fb1", "glow": "rgba(255, 143, 177, 0.45)", "depth": "Coral Garden"},
    "surprise": {"emoji": "⚡", "color": "#4be3ff", "glow": "rgba(75, 227, 255, 0.45)",  "depth": "Reef Break"},
    "sadness":  {"emoji": "💧", "color": "#5aa9e6", "glow": "rgba(90, 169, 230, 0.45)",  "depth": "Blue Current"},
    "fear":     {"emoji": "🦑", "color": "#8b7ec8", "glow": "rgba(139, 126, 200, 0.45)", "depth": "Twilight Zone"},
    "anger":    {"emoji": "🔥", "color": "#ff5d5d", "glow": "rgba(255, 93, 93, 0.45)",   "depth": "Thermal Vent"},
}

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_emotion(text: str, model, tokenizer):
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding="post")
    probs = model.predict(padded, verbose=0)[0]

    all_probabilities = {label: float(p) for label, p in zip(emotion_labels, probs)}
    predicted_emotion = max(all_probabilities, key=all_probabilities.get)
    confidence = all_probabilities[predicted_emotion]
    return predicted_emotion, confidence, all_probabilities


# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&family=Nunito+Sans:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif;
}

#MainMenu, header, footer { visibility: hidden; }

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 15% -5%, rgba(120, 240, 255, 0.10), transparent 60%),
        radial-gradient(ellipse 700px 500px at 90% 20%, rgba(120, 255, 220, 0.08), transparent 55%),
        linear-gradient(180deg, #030d1a 0%, #05263c 20%, #08415c 44%, #0a5a72 70%, #0c7a86 100%);
    background-attachment: fixed;
    color: #eaf6fb;
    overflow-x: hidden;
}

/* ---------- caustic light shafts ---------- */
.caustic-field {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    mix-blend-mode: screen;
}
.caustic-beam {
    position: absolute;
    top: -20%;
    width: 140px;
    height: 160%;
    background: linear-gradient(180deg, rgba(190, 245, 255, 0.10), rgba(190, 245, 255, 0) 70%);
    transform: rotate(9deg);
    animation: shimmer 9s ease-in-out infinite;
    filter: blur(6px);
}
@keyframes shimmer {
    0%, 100% { opacity: 0.35; transform: rotate(9deg) translateX(0); }
    50%      { opacity: 0.75; transform: rotate(6deg) translateX(30px); }
}

/* ---------- floating bubbles ---------- */
.bubble-field {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.bubble {
    position: absolute;
    bottom: -10%;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.55), rgba(255,255,255,0.05) 70%);
    animation: rise linear infinite;
    opacity: 0.5;
}
@keyframes rise {
    0%   { transform: translateY(0) translateX(0); opacity: 0; }
    10%  { opacity: 0.55; }
    100% { transform: translateY(-115vh) translateX(20px); opacity: 0; }
}

/* ---------- signature: drifting jellyfish ---------- */
.jelly-wrap {
    position: absolute;
    top: 8px;
    right: 6%;
    z-index: 1;
    animation: jelly-drift 10s ease-in-out infinite;
    opacity: 0.9;
}
.jelly-bell {
    width: 46px;
    height: 34px;
    border-radius: 50% 50% 45% 45%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.9), rgba(154, 235, 255, 0.55) 60%, rgba(154, 235, 255, 0.15) 100%);
    box-shadow: 0 0 22px 6px rgba(150, 235, 255, 0.45);
    animation: jelly-pulse 2.6s ease-in-out infinite;
}
.jelly-tentacles {
    display: flex;
    justify-content: center;
    gap: 4px;
    margin-top: -2px;
}
.jelly-tentacles span {
    width: 2px;
    height: 22px;
    border-radius: 2px;
    background: linear-gradient(180deg, rgba(180, 240, 255, 0.65), rgba(180, 240, 255, 0));
    animation: tentacle-sway 2.2s ease-in-out infinite;
}
.jelly-tentacles span:nth-child(2) { animation-delay: 0.2s; height: 26px; }
.jelly-tentacles span:nth-child(3) { animation-delay: 0.4s; height: 18px; }
.jelly-tentacles span:nth-child(4) { animation-delay: 0.6s; height: 24px; }
@keyframes jelly-drift {
    0%, 100% { transform: translate(0, 0); }
    50%      { transform: translate(-18px, 16px); }
}
@keyframes jelly-pulse {
    0%, 100% { transform: scaleX(1) scaleY(1); }
    50%      { transform: scaleX(0.88) scaleY(1.08); }
}
@keyframes tentacle-sway {
    0%, 100% { transform: rotate(-6deg); }
    50%      { transform: rotate(6deg); }
}

/* ---------- header ---------- */
.ocean-hero {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 2.2rem 0 0.6rem 0;
}
.ocean-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.68rem;
    color: #a9e9f2;
    font-weight: 700;
    margin-bottom: 0.7rem;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(150, 230, 245, 0.28);
}
.ocean-eyebrow .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4be3ff;
    box-shadow: 0 0 8px 2px rgba(75, 227, 255, 0.8);
    animation: dot-pulse 1.8s ease-in-out infinite;
}
@keyframes dot-pulse {
    0%, 100% { opacity: 0.5; transform: scale(0.85); }
    50%      { opacity: 1; transform: scale(1.15); }
}
.ocean-title {
    font-family: 'Baloo 2', sans-serif;
    font-weight: 800;
    font-size: 3.1rem;
    background: linear-gradient(100deg, #8fe9ff 0%, #ffffff 35%, #b6f3e8 65%, #8fe9ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
    line-height: 1.1;
    animation: title-sheen 6s ease-in-out infinite;
    text-shadow: 0 0 40px rgba(120, 230, 255, 0.25);
}
@keyframes title-sheen {
    0%, 100% { background-position: 0% center; }
    50%      { background-position: 100% center; }
}
.ocean-subtitle {
    color: #b7dde8;
    font-size: 0.98rem;
    margin-top: 0.6rem;
    font-weight: 400;
}

.wave-divider {
    position: relative;
    z-index: 1;
    line-height: 0;
    margin-top: 1.4rem;
}

/* ---------- glass card ---------- */
.glass-card {
    position: relative;
    z-index: 1;
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.14);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 1.6rem 1.6rem 1.2rem 1.6rem;
    margin-top: -0.4rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
}

.stTextArea textarea {
    background: rgba(4, 20, 33, 0.55) !important;
    border: 1px solid rgba(120, 220, 235, 0.35) !important;
    border-radius: 14px !important;
    color: #eaf6fb !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border: 1px solid #4be3ff !important;
    box-shadow: 0 0 0 1px #4be3ff !important;
}
.stTextArea label { color: #bfe9f2 !important; font-weight: 600; }

div.stButton > button {
    position: relative;
    overflow: hidden;
    width: 100%;
    background: linear-gradient(90deg, #0d7d92, #12b3c9, #0d7d92);
    background-size: 220% auto;
    color: #04141c;
    font-weight: 800;
    letter-spacing: 0.02em;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 0;
    font-size: 1.02rem;
    margin-top: 0.6rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-position 0.6s ease;
    animation: btn-glow 2.8s ease-in-out infinite;
}
@keyframes btn-glow {
    0%, 100% { box-shadow: 0 6px 18px rgba(18, 179, 201, 0.35); }
    50%      { box-shadow: 0 6px 28px rgba(18, 179, 201, 0.65); }
}
div.stButton > button:hover {
    transform: translateY(-2px);
    background-position: 100% center;
    box-shadow: 0 10px 26px rgba(18, 179, 201, 0.6);
    color: #04141c;
}
div.stButton > button:active {
    transform: translateY(0px) scale(0.99);
}

/* ---------- result pearl ---------- */
.pearl-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.8rem 0 0.6rem 0;
    position: relative;
    z-index: 1;
}
.pearl-stage {
    position: relative;
    width: 190px;
    height: 190px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.pearl-ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 1.5px dashed rgba(255,255,255,0.25);
    animation: ring-spin 14s linear infinite;
}
.pearl-ripple {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.55);
    animation: ripple-out 1.8s ease-out 1;
}
@keyframes ring-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes ripple-out {
    0%   { transform: scale(0.62); opacity: 0.9; }
    100% { transform: scale(1.35); opacity: 0; }
}
.pearl {
    width: 148px;
    height: 148px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'Baloo 2', sans-serif;
    animation: bob 3.2s ease-in-out infinite;
    border: 2px solid rgba(255,255,255,0.35);
    position: relative;
    z-index: 1;
}
@keyframes bob {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-9px); }
}
.pearl-emoji { font-size: 2.6rem; }
.pearl-label {
    text-transform: capitalize;
    font-weight: 800;
    font-size: 1.05rem;
    margin-top: 0.15rem;
    color: #04141c;
}
.pearl-depth {
    margin-top: 0.9rem;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9fe0ec;
    font-weight: 700;
}
.pearl-confidence {
    color: #eaf6fb;
    font-size: 0.9rem;
    margin-top: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ---------- depth gauge ---------- */
.depth-gauge {
    position: relative;
    z-index: 1;
    max-width: 480px;
    margin: 1.6rem auto 0.4rem auto;
    padding: 0.5rem 0.2rem 0 0.2rem;
}
.depth-track {
    position: relative;
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #ffd23f, #ff8fb1, #4be3ff, #5aa9e6, #8b7ec8, #ff5d5d);
    opacity: 0.45;
    margin: 0 8px;
}
.depth-stops {
    display: flex;
    justify-content: space-between;
    margin-top: -11px;
}
.depth-stop {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    width: 16.5%;
}
.depth-node {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
    border: 2px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}
.depth-stop.active .depth-node {
    transform: scale(1.35);
    box-shadow: 0 0 14px 4px var(--dot-glow);
}
.depth-stop-label {
    font-size: 0.6rem;
    text-align: center;
    color: rgba(234, 246, 251, 0.45);
    letter-spacing: 0.02em;
    line-height: 1.2;
}
.depth-stop.active .depth-stop-label {
    color: #eaf6fb;
    font-weight: 700;
}

/* ---------- probability bars ---------- */
.reef-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.55rem 0;
}
.reef-label {
    width: 92px;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: capitalize;
    color: #cdeef5;
    flex-shrink: 0;
}
.reef-track {
    flex-grow: 1;
    height: 14px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.09);
}
.reef-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
    background-image: linear-gradient(180deg, rgba(255,255,255,0.35), rgba(255,255,255,0) 60%);
    background-blend-mode: overlay;
    box-shadow: 0 0 8px 0 rgba(255,255,255,0.15) inset;
}
.reef-pct {
    width: 52px;
    text-align: right;
    font-size: 0.8rem;
    color: #b7dde8;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

.section-caption {
    text-align: center;
    color: #8fc9d6;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
    margin: 1.8rem 0 0.5rem 0;
}

.ocean-footer {
    text-align: center;
    color: rgba(183, 221, 232, 0.45);
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    margin: 2.4rem 0 1rem 0;
    position: relative;
    z-index: 1;
}
</style>

<div class="caustic-field">
<div class="caustic-beam" style="left:12%; animation-delay:0s;"></div>
<div class="caustic-beam" style="left:38%; animation-delay:2.5s; width:100px;"></div>
<div class="caustic-beam" style="left:66%; animation-delay:5s;"></div>
<div class="caustic-beam" style="left:85%; animation-delay:1.2s; width:90px;"></div>
</div>

<div class="bubble-field">
""" + "".join([
    f'<div class="bubble" style="left:{x}%; width:{w}px; height:{w}px; animation-duration:{d}s; animation-delay:{a}s;"></div>'
    for x, w, d, a in [
        (5, 14, 14, 0), (14, 8, 10, 2), (23, 18, 17, 1),
        (34, 10, 12, 4), (46, 22, 20, 0.5), (58, 9, 11, 3),
        (68, 16, 15, 2.5), (77, 11, 13, 5), (87, 20, 18, 1.5),
        (94, 8, 9, 4.5),
    ]
]) + """
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="ocean-hero">
    <div class="jelly-wrap">
        <div class="jelly-bell"></div>
        <div class="jelly-tentacles"><span></span><span></span><span></span><span></span></div>
    </div>
    <div class="ocean-eyebrow"><span class="dot"></span>BiGRU · Emotion Analysis</div>
    <h1 class="ocean-title">Current 🌊</h1>
    <p class="ocean-subtitle">Drop a sentence in and watch what emotion rises to the surface.</p>
</div>
<div class="wave-divider">
<svg viewBox="0 0 1440 90" xmlns="http://www.w3.org/2000/svg" style="width:100%; display:block;">
    <path fill="rgba(255,255,255,0.06)" d="M0,32 C240,80 480,0 720,24 C960,48 1200,88 1440,40 L1440,90 L0,90 Z"></path>
    <path fill="rgba(75,227,255,0.10)" d="M0,55 C240,20 480,90 720,55 C960,20 1200,70 1440,45 L1440,90 L0,90 Z"></path>
</svg>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT
# =========================================================
user_text = st.text_area(
    "What's on your mind?",
    placeholder="e.g. I can't believe we actually pulled off the demo in time...",
    height=120,
    max_chars=2000,
)

analyze_clicked = st.button("🌊 Dive In & Analyze", use_container_width=True)

# =========================================================
# INFERENCE + RESULTS
# =========================================================
if analyze_clicked:
    if not user_text.strip():
        st.warning("Drop a sentence in first — the water's still calm.")
    else:
        with st.spinner("Reading the current..."):
            try:
                model, tokenizer = load_artifacts()
                predicted_emotion, confidence, all_probabilities = predict_emotion(
                    user_text, model, tokenizer
                )
            except FileNotFoundError:
                st.error(
                    "Couldn't find the model or tokenizer in `artifacts/`. "
                    "Make sure `BiGRU.h5` and `tokenizer.pkl` are in place."
                )
                st.stop()

        meta = emotion_meta[predicted_emotion]

        st.markdown(f"""
        <div class="pearl-wrap">
            <div class="pearl-stage">
                <div class="pearl-ring"></div>
                <div class="pearl-ripple" style="border-color:{meta['color']};"></div>
                <div class="pearl" style="background: radial-gradient(circle at 35% 30%, {meta['color']}, {meta['color']}dd 70%); box-shadow: 0 0 55px 12px {meta['glow']};">
                    <div class="pearl-emoji">{meta['emoji']}</div>
                    <div class="pearl-label">{predicted_emotion}</div>
                </div>
            </div>
            <div class="pearl-depth">{meta['depth']}</div>
            <div class="pearl-confidence">{confidence * 100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # depth gauge — shallow to deep ordering
        depth_order = ["joy", "love", "surprise", "sadness", "fear", "anger"]
        stops = []
        for label in depth_order:
            m = emotion_meta[label]
            active = " active" if label == predicted_emotion else ""
            stops.append(
                f'<div class="depth-stop{active}" style="--dot-glow:{m["color"]};">'
                f'<div class="depth-node" style="{"background:" + m["color"] + ";" if label == predicted_emotion else ""}"></div>'
                f'<div class="depth-stop-label">{m["emoji"]}<br>{m["depth"]}</div></div>'
            )
        gauge_html = (
            '<div class="depth-gauge"><div class="depth-track"></div>'
            f'<div class="depth-stops">{"".join(stops)}</div></div>'
        )
        st.markdown(gauge_html, unsafe_allow_html=True)

        st.markdown('<div class="section-caption">Full reading</div>', unsafe_allow_html=True)

        sorted_probs = sorted(all_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        rows = []
        for label, prob in sorted_probs:
            color = emotion_meta[label]["color"]
            pct = prob * 100
            rows.append(
                f'<div class="reef-row"><div class="reef-label">{emotion_meta[label]["emoji"]} {label}</div>'
                f'<div class="reef-track"><div class="reef-fill" style="width:{pct}%; background:{color};"></div></div>'
                f'<div class="reef-pct">{pct:.1f}%</div></div>'
            )
        bars_html = "".join(rows)
        st.markdown(f'<div class="glass-card">{bars_html}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="ocean-footer">Bidirectional GRU emotion model · built by Surabhi</div>',
    unsafe_allow_html=True,
)
