import streamlit as st
import joblib
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ set page config must be first Streamlit command
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)

# NLTK setup
nltk_packages = ['punkt','stopwords','wordnet','omw-1.4']
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.pkl")

model = load_model()

# Preprocessing
punctuations = set(string.punctuation)
stop_words = set(stopwords.words('english')) - {'not','no','never'}
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

# Streamlit UI
st.title("🛒 Amazon Review Sentiment Analysis")
st.write("Enter an Amazon Product Review to Analyze its Sentiment")

user_review = st.text_area(
    "Review Text:",
    height=150,
    placeholder="Type your review here..."
)

if st.button("Analyze Sentiment"):
    if user_review.strip():
        with st.spinner("Analyzing..."):
            cleaned = preprocess_text(user_review)
            df = pd.DataFrame({'reviewText': [cleaned]})
            prediction = model.predict(df)[0]

            if prediction == 1:
                st.success("✅ **Result:** POSITIVE")
            else:
                st.error("❌ **Result:** NEGATIVE")
    else:
        st.warning("Please enter a review text.")

st.markdown("---")
st.caption("Model: Logistic Regression | Deployed with Streamlit")

