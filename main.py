import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import logging

# Load necessary models and tools
predictor = pickle.load(open("models/xgb.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
cv = pickle.load(open("models/cv.pkl", "rb"))

STOPWORDS = set(stopwords.words("english"))

st.title("Text Sentiment Predictor")

# Text input for sentiment prediction
text_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    try:
        corpus = []
        stemmer = PorterStemmer()
        review = re.sub("[^a-zA-Z]", " ", text_input)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

        # Transform the input text for prediction
        X_prediction = cv.transform(corpus).toarray()
        X_prediction_scl = scaler.transform(X_prediction)

        # Make prediction
        y_predictions = predictor.predict_proba(X_prediction_scl)
        print(y_predictions)
        y_predictions = y_predictions.argmax(axis=1)[0]

        # Display result
        sentiment = "Positive" if y_predictions == 1 else "Negative"
        st.write(f"Predicted sentiment: {sentiment}")

    except Exception as e:
        logging.error(f"Error in single prediction: {e}")
        st.write("Error in making prediction. Please try again.")
