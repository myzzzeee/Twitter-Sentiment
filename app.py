import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ----------------------------- Page setup -----------------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="wide")

st.markdown("""
<style>
/* ---- Cards & layout ---- */
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(14,165,164,0.15), rgba(99,102,241,0.10));
    border: 1px solid rgba(127,127,127,0.15);
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0 0 0.2rem 0; font-size: 1.7rem; }
.hero p  { margin: 0; opacity: 0.75; font-size: 0.95rem; }

.result-card {
    border-left: 6px solid var(--accent-color);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-top: 0.6rem;
    background: rgba(127, 127, 127, 0.08);
}
.result-card h4 { margin: 0 0 0.35rem 0; font-size: 0.85rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.03em; }
.result-card .label { font-size: 1.5rem; font-weight: 700; }
.result-card .conf { font-size: 0.85rem; opacity: 0.7; margin-top: 0.15rem; }

.example-chip {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(127,127,127,0.12);
    font-size: 0.82rem;
    margin: 0.15rem 0.3rem 0.15rem 0;
}

.stButton>button {
    background: #0ea5a4;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
    padding: 0.6rem 0;
}
.stButton>button:hover { background: #0c8988; color: white; }

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

SENTIMENT_STYLE = {
    "Positive":   {"color": "#22c55e", "emoji": "😊"},
    "Negative":   {"color": "#f43f5e", "emoji": "😠"},
    "Neutral":    {"color": "#0ea5e9", "emoji": "😐"},
    "Irrelevant": {"color": "#a855f7", "emoji": "🤷"},
}

LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]  # alphabetical, matches LabelEncoder

MODEL_INFO = {
    "SVC (TF-IDF + LinearSVC)": {
        "key": "svc",
        "desc": "A classic bag-of-words model: TF-IDF features (uni+bi-grams) feeding a linear support vector classifier. Fast and strong on clear-cut wording.",
    },
    "SimpleRNN (Bi-directional, deep learning)": {
        "key": "rnn",
        "desc": "A bidirectional SimpleRNN neural network with its own text vectorizer baked in. Learns word order and context, not just word presence.",
    },
}

EXAMPLE_TWEETS = [
    "This new update completely ruined the game, I'm furious.",
    "Just tried the new phone camera, absolutely stunning quality!",
    "The store closes at 9pm on weekdays.",
    "Watching a random cooking show, nothing special either way.",
]

# ----------------------------- Text cleaning (same as training) -----------------------------
# NOTE: the notebook keeps "!" and "?" in the cleaned text — they're used as a
# sentiment signal — so this must match exactly for both models (SVC and
# SimpleRNN) to get correctly-preprocessed input.
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s!?]', ' ', text)   # keep ! and ? — sentiment signal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------- Load models (cached) -----------------------------
@st.cache_resource
def load_svm():
    with open("sentiment_svm_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return pipeline, le

@st.cache_resource
def load_rnn():
    return tf.keras.models.load_model("sentiment_rnn_model.keras")

def svm_predict_all(pipeline, le, clean_text):
    """Returns a dict {label: probability} for every class."""
    scores = pipeline.decision_function([clean_text])[0]
    probs = np.exp(scores) / np.exp(scores).sum()
    return {cls: float(p) for cls, p in zip(le.classes_, probs)}

def rnn_predict_all(model, clean_text):
    """Returns a dict {label: probability} for every class."""
    probs = model.predict(tf.constant([clean_text]), verbose=0)[0]
    return {label: float(p) for label, p in zip(LABELS, probs)}

def render_result(label, probs):
    style = SENTIMENT_STYLE.get(label, {"color": "#888", "emoji": ""})
    confidence = probs[label]
    st.markdown(f"""
    <div class="result-card" style="--accent-color:{style['color']}">
        <h4>Prediction</h4>
        <div class="label" style="color:{style['color']}">{style['emoji']} {label}</div>
        <div class="conf">Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption("Probability across all classes")
    df = pd.DataFrame({
        "Sentiment": list(probs.keys()),
        "Probability": list(probs.values()),
    }).sort_values("Probability", ascending=True)
    st.bar_chart(df.set_index("Sentiment"), horizontal=True, height=180)

# ----------------------------- Session state -----------------------------
if "tweet_text" not in st.session_state:
    st.session_state.tweet_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

def set_example(text):
    st.session_state.tweet_text = text

# ----------------------------- Header -----------------------------
st.markdown("""
<div class="hero">
    <h1>💬 Tweet Sentiment Analyzer</h1>
    <p>Compare a classic ML model against a deep learning model on the same tweet.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_label = st.radio("Choose a model", list(MODEL_INFO.keys()))
    st.caption(MODEL_INFO[model_label]["desc"])

    st.divider()
    st.subheader("Try an example")
    for ex in EXAMPLE_TWEETS:
        st.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,), use_container_width=True)

    if st.session_state.history:
        st.divider()
        st.subheader("Recent history")
        for item in reversed(st.session_state.history[-5:]):
            style = SENTIMENT_STYLE.get(item["label"], {"emoji": ""})
            st.caption(f"{style['emoji']} **{item['label']}** ({item['model']}) — {item['text'][:40]}{'…' if len(item['text']) > 40 else ''}")

# ----------------------------- Main -----------------------------
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Enter a tweet")
    tweet = st.text_area(
        "Tweet text",
        height=140,
        placeholder="e.g. This new update completely ruined the game...",
        key="tweet_text",
        label_visibility="collapsed",
    )
    run = st.button("Analyze sentiment", type="primary")

with col_result:
    st.subheader("Result")
    if run:
        if not tweet.strip():
            st.warning("Please type a tweet first.")
        else:
            cleaned = clean_tweet(tweet)
            with st.spinner(f"Running {model_label}…"):
                try:
                    if MODEL_INFO[model_label]["key"] == "svc":
                        pipeline, le = load_svm()
                        probs = svm_predict_all(pipeline, le, cleaned)
                    else:
                        model = load_rnn()
                        probs = rnn_predict_all(model, cleaned)
                    label = max(probs, key=probs.get)
                    render_result(label, probs)
                    st.session_state.history.append({
                        "text": tweet.strip(),
                        "label": label,
                        "model": model_label.split(" ")[0],
                    })
                except (FileNotFoundError, OSError) as e:
                    st.error(f"Could not load the model file needed for **{model_label}**. ({e})")
    else:
        st.info("Type a tweet and click **Analyze sentiment** to see the prediction here.")
