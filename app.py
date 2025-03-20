import streamlit as st
import joblib

# Set page config
st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📈", layout="wide")

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model files not found! Ensure 'sentiment_model.pkl', 'tfidf_vectorizer.pkl', and 'label_encoder.pkl' are in the same directory.")
    st.stop()

# Prediction function
def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return label_encoder.inverse_transform(prediction)[0]

# Custom CSS for styling (fixed text area visibility)
st.markdown("""
    <style>
    .title {
        font-size: 40px !important;
        color: #1A73E8 !important;
        text-align: center !important;
        font-family: 'Arial', sans-serif !important;
        text-shadow: 2px 2px 4px #aaa !important;
    }
    .subtitle {
        font-size: 20px !important;
        color: #555 !important;
        text-align: center !important;
        margin-bottom: 30px !important;
    }
    .stButton>button {
        background-color: #1A73E8 !important;
        color: white !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #1557B0 !important;
        transform: scale(1.05) !important;
    }
    .stTextArea textarea {
        border: 2px solid #1A73E8 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        padding: 10px !important;
        background-color: #FFFFFF !important;  /* Changed to white for better visibility */
        color: #000000 !important;  /* Black text for contrast */
    }
    .stTextArea textarea::placeholder {
        color: #888 !important;  /* Placeholder text color */
    }
    .sidebar .sidebar-content {
        background-color: #F0F4F8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
    This AI-powered app analyzes the sentiment of news headlines or articles using an SVM model trained on news data.  
    - *Accuracy*: 75%  \n
    - *Classes*: Positive, Negative, Neutral\n 
    - *✅ Positive*: Optimistic and favorable news\n
    - *❌ Negetive News*: Concerning news\n
    - *⚖ Neutral News*: Balance and Factual news\n
    Made with ❤️ by: 
    - *Rishi Prasad Karan*\n
    - *Deepanshu Kumar Behera*\n
    - *Yashraj Bisoyi*\n
    - *Sarthak Dhal*\n
    - *Aryan Pattanayak*\n
    - *Sidharth Kiran Behera*\n
""")
st.sidebar.markdown("### Features 🚀")
st.sidebar.markdown("""
    - Real-time sentiment analysis  
    - Pre-trained model on news data  
    - Interactive UI with examples
""")

# Header
st.markdown('<h1 class="title">News Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Uncover the vibe of your news—Positive, Negative, or Neutral!</p>', unsafe_allow_html=True)

# Main content in two columns
col1, col2 = st.columns([2, 1])

with col1:
    # User input
    st.markdown("### Enter Your News Text")
    user_input = st.text_area("", placeholder="Type your news headline or article here...", height=200)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input)
            st.markdown("### Predicted Sentiment:")
            if sentiment == "positive":
                st.success(f"*{sentiment.capitalize()}* - Great news! 🚀")
            elif sentiment == "negative":
                st.error(f"*{sentiment.capitalize()}* - Not so good news. 📉")
            else:
                st.info(f"*{sentiment.capitalize()}* - Neutral vibes. ⚖️")
        else:
            st.warning("Please enter some text to analyze!")

with col2:
    # Example predictions (one button per example)
    st.markdown("### Try Examples")
    examples = [
        "The company reported a significant increase in revenue this quarter.",
        "Economic downturns have negatively impacted the stock market.",
        "Sales in Finland decreased by 2.0 % , and international sales decreased by 9.3 % in terms of euros , and by 15.1 % in terms of local currencies ."
    ]
    for example in examples:
        if st.button(f"Predict: '{example[:40]}...'", key=example):
            sentiment = predict_sentiment(example)
            st.write(f"*Text*: {example}")
            st.write(f"*Sentiment*: {sentiment.capitalize()} {'🚀' if sentiment == 'positive' else '📉' if sentiment == 'negative' else '⚖️'}")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #555; font-size: 14px;">Powered by Streamlit | Model Accuracy: 75% | © 2025 Rishi, Deepanshu, Yashraj, Sarthak, Aryan, Sidharth</p>', unsafe_allow_html=True)