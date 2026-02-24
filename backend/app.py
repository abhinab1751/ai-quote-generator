from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

model = None
tokenizer = None
index_to_word = None
max_sequence_len = None


def load_resources():
    global model, tokenizer, index_to_word, max_sequence_len

    if model is None:
        print("Loading model and resources...")

        model_path = os.path.join(BASE, "quote_model.h5")
        tokenizer_path = os.path.join(BASE, "tokenizer.pkl")
        config_path = os.path.join(BASE, "config.pkl")

        model = tf.keras.models.load_model(model_path, compile=False)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        max_sequence_len = config["max_sequence_len"]
        index_to_word = {i: w for w, i in tokenizer.word_index.items()}

        print("Model loaded successfully!")


def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_quote(seed_text, next_words=20, temperature=0.8):
    result = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        predicted = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted, temperature)

        output_word = index_to_word.get(predicted_index, "")

        if not output_word or output_word == "<OOV>":
            continue

        result += " " + output_word

    return result.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    try:
        load_resources()

        data = request.get_json() or {}
        topic = data.get("topic", "life").strip() or "life"
        count = max(1, min(int(data.get("count", 1)), 5))

        quotes = [generate_quote(topic) for _ in range(count)]

        return jsonify({
            "quotes": quotes,
            "topic": topic
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Generation failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)