"""
Spam Detection Web Application
Built with Streamlit - Deployed on Hugging Face Spaces
"""

import streamlit as st
import pickle
import re
import string
from datetime import datetime

# Page configuration (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    .spam-warning {
        background-color: #ff6b6b;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .ham-success {
        background-color: #51cf66;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        with open('models/spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.stop()

def clean_text(text):
    """Clean and preprocess text messages"""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_spam(message, model, vectorizer):
    """Predict if a message is spam or ham"""
    # Clean the message
    cleaned_message = clean_text(message)
    
    # Vectorize
    message_vectorized = vectorizer.transform([cleaned_message])
    
    # Predict
    prediction = model.predict(message_vectorized)[0]
    probability = model.predict_proba(message_vectorized)[0]
    
    return prediction, probability

# Main app
def main():
    # Header
    st.title("📧 SMS/Email Spam Detector")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This spam detector uses **Machine Learning** to identify spam messages.
        
        **Technologies used:**
        - TF-IDF Vectorization
        - Multinomial Naive Bayes
        - Streamlit for UI
        - Deployed on Hugging Face Spaces
        
        **Model Performance:**
        - Accuracy: ~98%
        - Precision: ~95%
        - Recall: ~97%
        """)
        
        st.header("📊 Statistics")
        # You can add real-time stats here
        
        st.header("💡 Tips")
        st.markdown("""
        Spam messages often contain:
        - Urgent language
        - Too many links
        - Misspellings
        - Requests for personal info
        """)
    
    # Main content area with two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter your message")
        
        # Text input area
        user_input = st.text_area(
            "Type or paste your message here:",
            height=200,
            placeholder="Example: Congratulations! You've won a $1000 gift card. Click here to claim your prize..."
        )
        
        # Predict button
        if st.button("🔍 Detect Spam", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing message..."):
                    # Load models
                    model, vectorizer = load_models()
                    
                    # Make prediction
                    prediction, probabilities = predict_spam(user_input, model, vectorizer)
                    
                    # Store in session state for persistence
                    st.session_state.prediction = prediction
                    st.session_state.probabilities = probabilities
                    st.session_state.message = user_input
                    st.session_state.timestamp = datetime.now()
            else:
                st.warning("⚠️ Please enter a message to analyze.")
    
    with col2:
        st.subheader("📈 Message Stats")
        if 'message' in st.session_state and st.session_state.message:
            msg = st.session_state.message
            st.metric("Message Length", f"{len(msg)} characters")
            st.metric("Word Count", f"{len(msg.split())} words")
            st.metric("Uppercase Count", f"{sum(1 for c in msg if c.isupper())}")
        else:
            st.info("Enter a message to see statistics")
    
    # Results section
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.subheader("🔎 Results")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if st.session_state.prediction == 1:
                st.markdown(f"""
                <div class="spam-warning">
                🚨 <b>SPAM DETECTED</b><br>
                This message has been classified as SPAM.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ham-success">
                ✅ <b>SAFE MESSAGE</b><br>
                This message appears to be legitimate (HAM).
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            prob_spam = st.session_state.probabilities[1]
            prob_ham = st.session_state.probabilities[0]
            
            st.markdown("### Confidence Scores")
            st.progress(float(prob_ham), text=f"Ham: {prob_ham:.2%}")
            st.progress(float(prob_spam), text=f"Spam: {prob_spam:.2%}")
        
        # Show cleaned message
        with st.expander("View processed message"):
            cleaned = clean_text(st.session_state.message)
            st.code(cleaned, language="text")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Built with Streamlit • Deployed on Hugging Face Spaces</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()