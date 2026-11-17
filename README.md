# 🛡️ NewsGuard AI - Fake News Detector

An end-to-end NLP project that classifies news articles as **Real** or **Fake**, deployed as a dark-themed Streamlit web app.

**Pipeline:** `Text → CountVectorizer (Bag-of-Words) → Logistic Regression → Real / Fake`

## ✨ Features
- Dark, professional UI with confidence ring and probability bars
- Explainable AI: shows the words that pushed the prediction towards Real or Fake
- One-click sample articles, low-confidence warnings, recent-check history
- Configurable label mapping (in case 0/1 encoding differs)

## 🧰 Tech Stack
Python · Scikit-learn · NumPy · Pandas · Streamlit

## 🚀 Run locally
```bash
git clone https://github.com/your-username/newsguard-ai.git
cd newsguard-ai
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure
```
newsguard-ai/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── .streamlit/config.toml
```

## ☁️ Deploy free
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → New app → select `app.py`.

## ⚠️ Disclaimer
A statistical model, not a fact-checker. Always verify important news with trusted sources.
