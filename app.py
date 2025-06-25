import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model_dl = load_model('fake_news_model.h5')

# Constants
MAX_LEN = 200  # Match training config

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Handle refresh
if "refresh_triggered" in st.session_state and st.session_state.refresh_triggered:
    st.session_state.input_text = ""
    st.session_state.refresh_triggered = False
    st.rerun()

# Title
st.title("📰 Fake News Detection App")


# Text input
st.text_area(
    "Enter the news article text:",
    key="input_text",
    height=200
)

# Buttons below input: left (Check), right (Refresh)
left_col, right_col = st.columns([1, 1])
with left_col:
    check = st.button("🔍 Check if it's Fake")
with right_col:
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.refresh_triggered = True
        st.rerun()

# Prediction logic
if check and st.session_state.input_text.strip():
    text = st.session_state.input_text.strip()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prediction = model_dl.predict(padded)
    label = "FAKE" if prediction[0][0] < 0.5 else "REAL"
    emoji = "🟥 FAKE" if label == "FAKE" else "🟩 REAL"

    st.subheader("🧠 Prediction Result:")
    st.success(f"This news is predicted to be **{emoji}**.")
elif check:
    st.warning("Please enter some news text.")
