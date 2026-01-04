import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index= imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

## load the model
model=load_model('simple_rnn_imdb.h5')

## Function to preprocess the input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [1]  # <START> token
    for word in words:
        encoded_review.append(word_index.get(word, 2))  # <UNK> = 2
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review

## Step 3- Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## Streamlit UI
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below and click **Predict** to classify sentiment.")

review_text = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="e.g. This movie was excellent and I loved it"
)

if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(review_text)

        if sentiment == "Positive":
            st.success(f"Sentiment: **{sentiment}** ")
        else:
            st.error(f"Sentiment: **{sentiment}** ")

        st.write(f"Prediction score: `{score:.4f}`")
