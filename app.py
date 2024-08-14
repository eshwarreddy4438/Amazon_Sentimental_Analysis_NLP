from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the model, scaler, and vectorizer
        predictor = pickle.load(open("models/rf.pkl", "rb"))
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        cv = pickle.load(open("models/cv.pkl", "rb"))
    except Exception as e:
        logging.error(f"Error loading model files: {e}")
        return jsonify({"error": "Model files not found or could not be loaded."})

    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")

            return response

        elif request.json and "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})
        else:
            return jsonify({"error": "Invalid input. Please provide text or a file."})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    try:
        corpus = []
        stemmer = PorterStemmer()
        review = re.sub("[^a-zA-Z]", " ", text_input)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)
        X_prediction = cv.transform(corpus).toarray()
        X_prediction_scl = scaler.transform(X_prediction)
        y_predictions = predictor.predict_proba(X_prediction_scl)
        y_predictions = y_predictions.argmax(axis=1)[0]

        return "Positive" if y_predictions == 1 else "Negative"
    except Exception as e:
        logging.error(f"Error in single prediction: {e}")
        return "Error"


def bulk_prediction(predictor, scaler, cv, data):
    try:
        corpus = []
        stemmer = PorterStemmer()
        for i in range(data.shape[0]):
            review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
            review = review.lower().split()
            review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
            review = " ".join(review)
            corpus.append(review)

        X_prediction = cv.transform(corpus).toarray()
        X_prediction_scl = scaler.transform(X_prediction)
        y_predictions = predictor.predict_proba(X_prediction_scl)
        y_predictions = y_predictions.argmax(axis=1)
        y_predictions = list(map(sentiment_mapping, y_predictions))

        data["Predicted sentiment"] = y_predictions
        predictions_csv = BytesIO()
        data.to_csv(predictions_csv, index=False)
        predictions_csv.seek(0)

        graph = get_distribution_graph(data)

        return predictions_csv, graph
    except Exception as e:
        logging.error(f"Error in bulk prediction: {e}")
        return None, None


def get_distribution_graph(data):
    try:
        fig = plt.figure(figsize=(5, 5))
        colors = ("green", "red")
        wp = {"linewidth": 1, "edgecolor": "black"}
        tags = data["Predicted sentiment"].value_counts()
        explode = (0.01, 0.01)

        tags.plot(
            kind="pie",
            autopct="%1.1f%%",
            shadow=True,
            colors=colors,
            startangle=90,
            wedgeprops=wp,
            explode=explode,
            title="Sentiment Distribution",
            xlabel="",
            ylabel="",
        )

        graph = BytesIO()
        plt.savefig(graph, format="png")
        plt.close()
        return graph
    except Exception as e:
        logging.error(f"Error generating distribution graph: {e}")
        return BytesIO()


def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)

# Testing single_prediction
print(single_prediction(predictor, scaler, cv, "I love this product!"))

# Testing bulk_prediction
test_data = pd.DataFrame({"Sentence": ["I love this product!", "I hate this product!"]})
csv_file, graph = bulk_prediction(predictor, scaler, cv, test_data)
print(csv_file.getvalue())  # Should print the CSV content
