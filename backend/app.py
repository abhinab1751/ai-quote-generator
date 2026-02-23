from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os


app = Flask(__name__)
CORS(app)


BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, "quote_model.h5")
TOKENIZER_PATH = os.path.join(BASE, "tokenizer.pkl")
CONFIG_PATH = os.path.join(BASE, "config.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)

max_sequence_len = config["max_sequence_len"]

index_to_word = {i: w for w, i in tokenizer.word_index.items()}



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

        output_word = index_to_word.get(predicted_index, "<UNK>")

        if output_word in ["<UNK>", "", "<OOV>"]:
            continue

        result += " " + output_word

    return result.strip()



@app.route("/")
def home():
    return "AI Quote Generator Backend Running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


@app.route("/generate", methods=["POST"])
def generate():

    try:
        data = request.get_json()

        topic = data.get("topic", "life").strip()
        if not topic:
            topic = "life"

        count = int(data.get("count", 1))
        count = max(1, min(count, 5))

        quotes = [
            generate_quote(
                seed_text=topic,
                next_words=20,
                temperature=0.8
            )
            for _ in range(count)
        ]

        return jsonify({
            "topic": topic,
            "quotes": quotes
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500



if __name__ == "__main__":
    app.run()