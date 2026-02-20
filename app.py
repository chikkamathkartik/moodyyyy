import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(
    page_title="Moodyyyy",
    page_icon="🎭",
    layout="centered"
)

# ---- Load Model ----
# @st.cache_resource means: load the model ONCE and reuse it
# Without this, the model would reload every time the user types — very slow
@st.cache_resource # load once and reuse again
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

classifier = load_model()

# ---- Emoji map for each emotion ----
emotion_emojis = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐" # can use custom made emojis and pictures.
}

# ---- UI ----
st.title("🎭 Moodyyyy")
st.subheader("Real Time Emotion Detection from Text")
st.write("Type any sentence below and I will tell you what emotion it has.")

# Text input box
user_input = st.text_area("Enter your text here:", placeholder="e.g. I can't believe how amazing today was!")

# Analyze button
if st.button("Analyze Emotion"):

    if user_input.strip() == "": # checks if empty spaces in the text area
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            results = classifier(user_input)[0]

            # Sort emotions by score highest to lowest
            results = sorted(results, key=lambda x: x['score'], reverse=True)

            # Top emotion
            top = results[0]
            top_label = top['label'].capitalize()
            top_score = round(top['score'] * 100, 2)
            top_emoji = emotion_emojis.get(top['label'], "")

            st.markdown("---")
            st.markdown(f"### Detected Emotion: {top_emoji} **{top_label}**")
            st.markdown(f"**Confidence:** {top_score}%")
            st.markdown("---")

            # Bar chart of all emotions
            st.markdown("#### Emotion Breakdown")
            labels = [r['label'].capitalize() for r in results]
            scores = [round(r['score'] * 100, 2) for r in results]
            colors = ["#4CAF50" if r['label'] == top['label'] else "#90CAF9" for r in results]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(labels, scores, color=colors)
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            ax.invert_yaxis()

            # Add score labels on bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{score}%", va='center', fontsize=10)

            st.pyplot(fig)

# ---- Footer ----
st.markdown("---")
st.caption("Built with HuggingFace Transformers + PyTorch + Streamlit")