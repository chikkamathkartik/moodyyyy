---
title: Moodyyyy
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🎭 Moodyyyy - Real Time Emotion Detection

A web application that detects emotions in text using a fine tuned Transformer model, built with HuggingFace Transformers, PyTorch, and Streamlit.

##  Live Demo
[Click here to try it live](https://huggingface.co/spaces/haiku123412/Moodyyyy) <!-- We'll add the real link after deployment tomorrow -->

##  How It Works
1. User enters any text in the input box
2. The app passes the text through a fine-tuned **DistilRoBERTa** model
3. The model returns confidence scores for 7 emotions: joy, sadness, anger, fear, surprise, disgust, and neutral
4. Results are displayed with a confidence bar chart

##  Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| HuggingFace Transformers | Pre-trained emotion detection model |
| PyTorch | Deep learning backend |
| Streamlit | Web app UI |
| Matplotlib | Confidence bar chart visualization |

##  Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/moodlens.git
cd moodlens

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

##  Model Details
- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Architecture:** DistilRoBERTa (distilled version of RoBERTa, based on BERT)
- **Approach:** Transfer Learning — using a model pre-trained and fine-tuned on emotion datasets
- **Emotions Detected:** joy, sadness, anger, fear, surprise, disgust, neutral

##  Author
Your Name — [LinkedIn](#) | [GitHub](#)