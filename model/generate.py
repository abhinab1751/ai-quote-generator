import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import random
import pickle 

model = tf.keras.models.load_model("quote_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
max_sequence_len = config["max_sequence_len"]

index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_quote(seed_text, next_words=20, temperature=0.8):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predicted = model.predict(token_list, verbose=0)[0]

        predicted_index = sample_with_temperature(predicted, temperature)

        output_word = index_to_word.get(predicted_index, "")

        if output_word == "" or output_word == "<OOV>": 
            continue

        seed_text += " " + output_word

    return seed_text


print("Creative (temp=1.2):")
print(generate_quote("life is", temperature=1.2))

print("\nBalanced (temp=0.8):")
print(generate_quote("life is", temperature=0.8))

print("\nConservative (temp=0.5):")
print(generate_quote("life is", temperature=0.5))