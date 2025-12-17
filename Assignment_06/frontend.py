import streamlit as st
import requests
API_URL = "http://127.0.0.1:8001/predict"
st.set_page_config(
    page_title = "Amazon Review Sentiment Analysis",
    page_icon = "🛒",
    layout = "centered"
)
# App Header
st.title("🛒 Amazon Review Sentiment Analysis")
st.write("Enter an Amazon Product Review to Analyze its Sentiment (Positive/Negative)")

# Input 
user_review = st.text_area("Review Text:", height = 150, placeholder = "Type your review here...")

#button
if st.button("Analyze Sentiment"):
    if user_review.strip():
        with st.spinner("Analyzing..."):
            try:
                payload = {'reviewText': user_review}
                response = requests.post(API_URL, json = payload)
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment']
                    #st.devider()
                    
                    if sentiment == "Positive":
                        st.success(f"✅ **Result:** The review is **POSITIVE**")
                    else:
                        st.error(f"❌ **Result:** The review is **NEGATIVE**")
                else:
                    st.error(f"Error: Server returned status code {response.status_code}")
            except requests.exceptions.RequestException:
                st.error("Error: Unable to connect to the server. Please try again later.")
                st.warning(f"Make sure the backend API is runing at {API_URL}")
    else:
        st.warning("Please enter a review text to analyze.")


st.markdown("---")
st.caption("Backend powered by FastAPI | Model: Logistic Regression")
