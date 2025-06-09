
import streamlit as st
import pickle
import numpy as np
import os
if not os.path.exists("fake_news_model.pkl"):
    st.error("Model file not found. Please upload it.")
else:
    model = pickle.load(open("fake_news_model.pkl", "rb"))

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit app layout
st.title("📰 Fake News Classifier")
st.subheader("Paste a news article below to check if it's real or fake:")

# Input text
input_text = st.text_area("News Article", height=200)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([input_text])
        prediction = model.predict(input_vec)
        probability = model.predict_proba(input_vec)[0]

        # Show probabilities
        st.write(f"🧠 Model Confidence:")
        st.write(f"👉 Real News: {probability[1] * 100:.2f}%")
        st.write(f"👉 Fake News: {probability[0] * 100:.2f}%")

        # Final classification
        if prediction[0] == 1:
            st.success("✅ This news is REAL.")
        else:
            st.error("❌ This news is FAKE.")
