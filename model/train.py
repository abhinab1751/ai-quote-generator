import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import json
import os
import pickle


file_path = os.path.join("..", "data", "quotes.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

lines = []

for item in data:
    if "Quote" in item:
        quote = item["Quote"].strip()
        if quote:
            lines.append(quote)

print("Total lines loaded:", len(lines))


lines = lines[:8000]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

tokenizer.word_index = {
    w: i for w, i in tokenizer.word_index.items()
    if tokenizer.word_counts[w] >= 2
}
tokenizer.word_index = {w: i+1 for i, (w, _) in enumerate(tokenizer.word_index.items())}

total_words = len(tokenizer.word_index) + 1
print("Total vocabulary size:", total_words)


input_sequences = []

for line in lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    token_list = token_list[:26]  
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print("Total sequences:", len(input_sequences))


max_sequence_len = 25

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len + 1, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64),
    tf.keras.layers.LSTM(128),                         
    tf.keras.layers.Dropout(0.2),                       
    tf.keras.layers.Dense(total_words, activation="softmax")
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
    metrics=['accuracy']
)

model.build(input_shape=(None, max_sequence_len))

model.summary()


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-5),
]

history = model.fit(
    X,
    y,
    epochs=30,         
    batch_size=512,     
    verbose=1,
    callbacks=callbacks
)


model.save("quote_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("config.pkl", "wb") as f:
    pickle.dump({"max_sequence_len": max_sequence_len}, f)

print("Model saved successfully!")