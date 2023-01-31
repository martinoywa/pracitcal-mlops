from flask import Flask, request, jsonify
from transformers import pipeline


app = Flask(__name__)

LABELS = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}


def output_formatter(list):
    label = LABELS[list[0]["label"]]
    confidence = list[0]["score"]
    return {
        "label": label,
        "confidence": confidence
    }

@app.route("/api/v1")
def home():
    return "Welcome to the sentiment API."


@app.route("/api/v1/sentiment")
def predict_sentiment():
    pipe = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
    input = request.args.get("input")
    pred = pipe(input)

    return jsonify({"prediction": output_formatter(pred)})


if __name__ == "__main__":
    app.run(debug=True)
