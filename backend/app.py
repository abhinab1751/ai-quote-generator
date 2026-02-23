from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})


BASE = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE, "quote_model.h5"))

with open(os.path.join(BASE, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(BASE, "config.pkl"), "rb") as f:
    config = pickle.load(f)

max_sequence_len = config["max_sequence_len"]
index_to_word = {i: w for w, i in tokenizer.word_index.items()}


def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_quote(seed_text, next_words=20, temperature=0.8):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        predicted = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted, temperature)
        output_word = index_to_word.get(predicted_index, "")
        if output_word == "" or output_word == "<OOV>":
            continue
        result += " " + output_word
    return result.strip()


@app.route("/")
def home():
    return send_from_directory(BASE, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json() or {}

    topic = data.get("topic", "life").strip() or "life"
    count = max(1, min(int(data.get("count", 1)), 5))

    quotes = [generate_quote(topic, 20, 0.8) for _ in range(count)]

    return jsonify({"quotes": quotes, "topic": topic})


if __name__ == "__main__":
    app.run(debug=True, port=5000)