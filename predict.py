from transformers import pipeline

# Load  pre-trained emotion detection model from HuggingFace
 
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # Return scores for ALL emotions, not just the top 1
)

# a sample sentence
text = "I just got my dream job offer and I can't stop smiling!"

results = emotion_classifier(text)

# Print the results
print(f"\nText: {text}\n")
for item in results[0]:
    label = item['label']
    score = round(item['score'] * 100, 2)
    print(f"  {label}: {score}%")