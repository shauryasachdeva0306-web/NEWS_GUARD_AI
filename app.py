"""
NewsGuard AI - Fake News Detection Web App
------------------------------------------
Pipeline: Raw news text -> CountVectorizer (Bag-of-Words) -> Logistic Regression -> Real / Fake
Run with:  streamlit run app.py
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsGuard AI | Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}

.hero {
    padding: 2.2rem 2.4rem; border-radius: 20px; margin-bottom: 1.5rem;
    background: radial-gradient(1200px 400px at 0% 0%, rgba(0,229,168,.16), transparent 60%),
                radial-gradient(900px 400px at 100% 100%, rgba(99,102,241,.18), transparent 60%),
                #111827;
    border: 1px solid rgba(255,255,255,.07);
}
.hero h1 {font-size: 2.6rem; margin: 0 0 .3rem 0; font-weight: 800; letter-spacing: -.5px;}
.hero h1 span {background: linear-gradient(90deg,#00E5A8,#6EA8FF); -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;}
.hero p {color: #9AA6BD; font-size: 1.05rem; margin: 0;}
.badge {display:inline-block; padding: 4px 12px; margin: 14px 8px 0 0; border-radius: 999px;
        font-size: .78rem; color:#B8C2D9; background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.09);}

.card {background:#131A2A; border:1px solid rgba(255,255,255,.07); border-radius:16px;
       padding: 1.3rem 1.5rem; margin-bottom: 1rem;}
.card h4 {margin: 0 0 .6rem 0; color:#E6EAF2;}

.verdict {border-radius: 18px; padding: 1.6rem 1.8rem; display:flex; align-items:center; gap: 1.6rem;
          border: 1px solid; margin-bottom: 1rem;}
.verdict.real {background: linear-gradient(135deg, rgba(0,229,168,.14), rgba(0,229,168,.03));
               border-color: rgba(0,229,168,.45);}
.verdict.fake {background: linear-gradient(135deg, rgba(255,77,109,.16), rgba(255,77,109,.03));
               border-color: rgba(255,77,109,.5);}
.verdict .label {font-size: .8rem; letter-spacing: 2px; color:#9AA6BD; text-transform: uppercase;}
.verdict .title {font-size: 2rem; font-weight: 800; margin: .1rem 0;}
.verdict.real .title {color:#00E5A8;} .verdict.fake .title {color:#FF4D6D;}
.verdict .sub {color:#9AA6BD; font-size:.95rem;}

.ring {--p: 50; --c:#00E5A8; width: 112px; height:112px; border-radius:50%; flex-shrink:0;
       background: conic-gradient(var(--c) calc(var(--p)*1%), rgba(255,255,255,.08) 0);
       display:grid; place-items:center;}
.ring::before {content:""; position:absolute; width: 88px; height: 88px; border-radius:50%; background:#0F1626;}
.ring span {position:relative; font-weight:800; font-size:1.35rem;}

.bar-wrap {margin: .5rem 0 .9rem 0;}
.bar-top {display:flex; justify-content:space-between; font-size:.88rem; color:#B8C2D9; margin-bottom:4px;}
.bar {height: 10px; border-radius: 99px; background: rgba(255,255,255,.07); overflow:hidden;}
.bar > div {height:100%; border-radius:99px;}

.chip {display:inline-block; padding: 4px 11px; margin: 3px 4px 3px 0; border-radius: 8px;
       font-size:.85rem; font-family: ui-monospace, Menlo, monospace;}
.chip.fake {background: rgba(255,77,109,.14); color:#FF8FA3; border:1px solid rgba(255,77,109,.35);}
.chip.real {background: rgba(0,229,168,.12); color:#52F0C4; border:1px solid rgba(0,229,168,.35);}

.step {display:flex; gap:14px; align-items:flex-start; margin-bottom: 14px;}
.step .n {min-width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-weight:800;
          background: rgba(0,229,168,.14); color:#00E5A8; border:1px solid rgba(0,229,168,.35);}
.step b {color:#E6EAF2;} .step div {color:#9AA6BD; font-size:.93rem;}

.stButton > button {border-radius: 12px; font-weight: 700; padding: .6rem 1.4rem;}
.stTextArea textarea {background:#0F1626; border-radius: 12px; font-size: 1rem;}
.footer {text-align:center; color:#6B7791; font-size:.85rem; padding: 1.5rem 0 .5rem 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load artifacts (cached so they load only once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    feature_names = np.array(vectorizer.get_feature_names_out())
    return model, vectorizer, feature_names


try:
    model, vectorizer, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("`model.pkl` or `vectorizer.pkl` not found. Keep them in the same folder as `app.py`.")
    st.stop()

# ----------------------------------------------------------------------------
# Sample articles
# ----------------------------------------------------------------------------
SAMPLES = {
    "📰 Policy report": (
        "WASHINGTON (Reuters) - The U.S. Senate voted on Tuesday to advance a bipartisan infrastructure "
        "spending bill, with lawmakers saying the measure would fund road, bridge and broadband projects "
        "over the next five years. The vote followed weeks of negotiations between party leaders, and a "
        "final vote is expected later this week, a senior aide told reporters."
    ),
    "🚨 Viral claim": (
        "BREAKING!!! You won't believe what they are hiding from you! Anonymous insiders reveal that "
        "the mainstream media has been LYING about everything. Share this before it gets deleted! "
        "The elites don't want you to know this shocking secret that will change your life forever."
    ),
}

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("news_text", "")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ NewsGuard AI")
    st.caption("Fake news detection using NLP & Machine Learning")
    st.divider()

    st.markdown("### Model Info")
    st.markdown(
        f"""
- **Algorithm:** Logistic Regression  
- **Features:** Bag-of-Words (CountVectorizer)  
- **Vocabulary size:** `{len(feature_names):,}`  
        """
    )
    st.divider()

    with st.expander("⚙️ Advanced settings"):
        positive_label = st.radio(
            "Model output `1` means:",
            ["Real", "Fake"],
            index=0,
            help="Depends on how you encoded labels while training. Switch if results look inverted.",
        )
        threshold = st.slider("Low-confidence warning below", 50, 90, 65, 1, format="%d%%")

    st.markdown("### 🕘 Recent checks")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            icon = "✅" if item["label"] == "Real" else "🚫"
            st.markdown(f"{icon} **{item['label']}** · {item['conf']:.0f}%  \n<small>{item['snippet']}</small>",
                        unsafe_allow_html=True)
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No analyses yet.")

# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def analyze(text: str, positive_label: str):
    """Vectorize -> predict -> explain."""
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p1 = proba[classes.index(1)] if 1 in classes else proba[-1]
    p_real = p1 if positive_label == "Real" else 1 - p1
    p_fake = 1 - p_real

    # Per-word contribution = word count x model coefficient
    coefs = model.coef_[0]
    counts = X.toarray()[0]
    contrib = counts * coefs
    # Convert so that positive value => pushes toward REAL
    if positive_label == "Fake":
        contrib = -contrib
    idx = np.nonzero(counts)[0]
    words = [(feature_names[i], contrib[i]) for i in idx]
    real_words = sorted([w for w in words if w[1] > 0], key=lambda x: -x[1])[:8]
    fake_words = sorted([w for w in words if w[1] < 0], key=lambda x: x[1])[:8]
    return p_real, p_fake, real_words, fake_words, int(len(idx))


def bar(label, pct, color):
    return f"""<div class="bar-wrap"><div class="bar-top"><span>{label}</span><span>{pct:.1f}%</span></div>
    <div class="bar"><div style="width:{pct}%; background:{color};"></div></div></div>"""

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🛡️ NewsGuard <span>AI</span></h1>
  <p>Paste any news article or headline and let machine learning tell you if it looks real or fake.</p>
  <span class="badge">Python</span><span class="badge">Scikit-learn</span>
  <span class="badge">NLP</span><span class="badge">Logistic Regression</span>
  <span class="badge">Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_check, tab_how, tab_about = st.tabs(["🔍 Analyze News", "⚙️ How it works", "👤 About"])

# ----------------------------------------------------------------------------
# Tab 1 - Analyzer
# ----------------------------------------------------------------------------
with tab_check:
    st.markdown("**Try an example:**")
    cols = st.columns(len(SAMPLES) + 1)
    for col, (name, text) in zip(cols, SAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.news_text = text
    if cols[-1].button("🧹 Clear", use_container_width=True):
        st.session_state.news_text = ""

    news = st.text_area(
        "News article",
        key="news_text",
        height=220,
        placeholder="Paste the full news article or headline here (longer text gives more reliable results)...",
        label_visibility="collapsed",
    )
    word_count = len(news.split())
    st.caption(f"{word_count} words · {len(news)} characters")

    go = st.button("🔎 Analyze News", type="primary", use_container_width=True)

    if go:
        if word_count < 5:
            st.warning("Please enter at least a few words (ideally a full paragraph) for a meaningful prediction.")
        else:
            with st.spinner("Vectorizing text and running the model..."):
                p_real, p_fake, real_words, fake_words, known = analyze(news, positive_label)

            is_real = p_real >= 0.5
            conf = max(p_real, p_fake) * 100
            kind = "real" if is_real else "fake"
            color = "#00E5A8" if is_real else "#FF4D6D"
            title = "✅ Likely REAL News" if is_real else "🚫 Likely FAKE News"
            sub = ("The writing patterns match credible reporting." if is_real
                   else "The writing patterns match misleading or fabricated content.")

            st.session_state.history.append(
                {"label": "Real" if is_real else "Fake", "conf": conf,
                 "snippet": (news[:48] + "...") if len(news) > 48 else news,
                 "time": datetime.now().strftime("%H:%M")}
            )

            st.markdown(
                f"""<div class="verdict {kind}">
                      <div class="ring" style="--p:{conf:.0f}; --c:{color}"><span>{conf:.0f}%</span></div>
                      <div><div class="label">Prediction · confidence</div>
                      <div class="title">{title}</div><div class="sub">{sub}</div></div>
                    </div>""",
                unsafe_allow_html=True,
            )

            if conf < threshold:
                st.warning("⚠️ Low confidence. The model is unsure. Verify this story with trusted sources.")
            if known < 5:
                st.info("Very few words were recognised by the model's vocabulary, so treat this result with caution.")

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.markdown('<div class="card"><h4>📊 Probability breakdown</h4>', unsafe_allow_html=True)
                st.markdown(bar("Real", p_real * 100, "#00E5A8") + bar("Fake", p_fake * 100, "#FF4D6D"),
                            unsafe_allow_html=True)
                st.markdown(f"<small style='color:#6B7791'>{known} unique words matched the model vocabulary.</small></div>",
                            unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card"><h4>🧠 Why this prediction?</h4>', unsafe_allow_html=True)
                st.markdown("<small style='color:#9AA6BD'>Words that pushed the model most strongly:</small>",
                            unsafe_allow_html=True)
                st.markdown("**Towards Real**", help="Words that increased the 'Real' score")
                st.markdown("".join(f'<span class="chip real">{w}</span>' for w, _ in real_words) or "—",
                            unsafe_allow_html=True)
                st.markdown("**Towards Fake**")
                st.markdown("".join(f'<span class="chip fake">{w}</span>' for w, _ in fake_words) or "—",
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.caption("⚠️ This is a statistical model, not a fact-checker. Always cross-check important news with trusted sources.")

# ----------------------------------------------------------------------------
# Tab 2 - How it works
# ----------------------------------------------------------------------------
with tab_how:
    st.markdown('<div class="card"><h4>The ML pipeline</h4>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="step"><div class="n">1</div><div><b>Input</b><br>The user pastes a news article or headline.</div></div>
<div class="step"><div class="n">2</div><div><b>Vectorization</b><br>A <code>CountVectorizer</code> lowercases the text, splits it into words and converts it into a numeric Bag-of-Words vector.</div></div>
<div class="step"><div class="n">3</div><div><b>Classification</b><br>A <code>Logistic Regression</code> model computes the probability that the article is real or fake.</div></div>
<div class="step"><div class="n">4</div><div><b>Explainability</b><br>Each word's count is multiplied by its learned coefficient, showing which words influenced the decision.</div></div>
</div>""",
        unsafe_allow_html=True,
    )
    st.code("text → CountVectorizer.transform() → LogisticRegression.predict_proba() → Real / Fake", language="text")

# ----------------------------------------------------------------------------
# Tab 3 - About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
<div class="card"><h4>About NewsGuard AI</h4>
<p style="color:#9AA6BD">NewsGuard AI is an end-to-end machine learning project that detects misinformation using
Natural Language Processing. It covers data preparation, model training, model persistence and deployment
as an interactive web app.</p>
<p><b>Tech stack:</b> Python · Pandas · NumPy · Scikit-learn · Streamlit</p>
<p><b>Built by:</b> <i>Your Name</i> ·
<a href="https://www.linkedin.com/in/your-profile" target="_blank">LinkedIn</a> ·
<a href="https://github.com/your-username/newsguard-ai" target="_blank">GitHub</a></p>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">NewsGuard AI · Built with Python & Streamlit</div>', unsafe_allow_html=True)
