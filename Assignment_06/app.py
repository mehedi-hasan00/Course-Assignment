from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
import pickle
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk_packages = ['punkt','stopwords','wordnet','omw-1.4']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)


# load model
try:
    model = joblib.load('sentiment_pipeline.pkl')
except Exception as e:
    print(f"Error Loading Model: {e}")
    raise e
#preprocessing 
punctuations = set(string.punctuation)
stopwords = set(stopwords.words('english')) - {'not','no','never'}

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = ''.join([c for c in text if c not in punctuations])
    words = text.split()
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stopwords]
    return ' '.join(words)

app = FastAPI(title="Amazon Sentiment Analysis API")

class ReviewRequest(BaseModel):
    reviewText: Annotated[str, Field(..., example="The product was excellent!")]

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}

@app.post('/predict')
def predict_sentiment(data:ReviewRequest):
    try:
        cleaned_text = preprocess_text(data.reviewText)
        df = pd.DataFrame({'reviewText': [cleaned_text]})
        prediction = model.predict(df)[0]
        result = "Positive" if prediction == 1 else "Negative"

        return {
            "original_review": data.reviewText,
            "cleaned_review": cleaned_text, # ডিবাগিংয়ের জন্য ক্লিন টেক্সটও রিটার্ন করলাম
            "sentiment": result,
            "label": int(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
