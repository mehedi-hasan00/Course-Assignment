import streamlit as st
import joblib
import string
import nltk
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)


# NLTK Setup
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

packages = [
    ('punkt', 'tokenizers/punkt'),
    ('stopwords', 'corpora/stopwords'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4')
]

for name, path in packages:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, download_dir=nltk_data_path)


@st.cache_resource
def load_model():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, "sentiment_pipeline.pkl")
    
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'sentiment_pipeline.pkl' file not found. Please make sure it is in the same folder.")
    st.stop()


# Preprocessing

punctuations = set(string.punctuation)
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = ''.join([c for c in text if c not in punctuations])
    words = text.split()
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words
    ]
    return ' '.join(words)
# app header
st.title("🛒 Amazon Review Sentiment Analysis")
st.write("Enter an Amazon Product Review to Analyze its Sentiment")

user_review = st.text_area(
    "Review Text:",
    height=150,
    placeholder="Type your review here... (e.g., This product is amazing!)"
)

if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip():
        with st.spinner("Analyzing..."):
            try:
                # Preprocess the text
                cleaned = preprocess_text(user_review)
                
                df = pd.DataFrame({'reviewText': [cleaned]})
                
                # Predict
                prediction = model.predict(df)[0]

                st.divider()
                if prediction == 1:
                    st.success("✅ **Result:**  **POSITIVE** Review! 😊")
                else:
                    st.error("❌ **Result:**  **NEGATIVE** Review! 😞")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a review text first.")

st.markdown("---")
st.caption("Model: Logistic Regression")