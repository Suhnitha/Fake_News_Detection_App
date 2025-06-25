from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load ML model (e.g., Logistic Regression)
with open('model.pkl', 'rb') as f:
    model_ml = pickle.load(f)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load deep learning model
model_dl = load_model('fake_news_model.h5')

# Define max length used during training
MAX_LEN = 300

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Predict
    prob = model_dl.predict(padded_seq)[0][0]
    result = "FAKE" if prob < 0.5 else "REAL"

    return render_template('index.html', prediction=result, probability=f"{prob:.2f}")

if __name__ == '__main__':
    app.run(debug=True)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
