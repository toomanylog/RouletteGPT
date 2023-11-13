import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_and_prepare_data(file_path, sequence_length=10):
    data = np.genfromtxt(file_path, delimiter='\n', dtype=int)
    data = data[(data >= 0) & (data <= 36)]
    sequences = np.array([data[i:i + sequence_length] for i in range(len(data) - sequence_length)])
    targets = data[sequence_length:]
    return sequences, targets

sequences, targets = load_and_prepare_data('data.txt')

train_index = int(0.8 * len(sequences))
train_data, train_targets = sequences[:train_index], targets[:train_index]
val_data, val_targets = sequences[train_index:], targets[train_index:]

model = Sequential([
    Embedding(input_dim=37, output_dim=50),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(37, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=50, batch_size=16)

last_sequence = val_data[-1].reshape(1, -1)
prediction = model.predict(last_sequence)
predicted_number = np.argmax(prediction, axis=1)[0]

print(f"Numéro prédit : {predicted_number}")

input("Appuyez sur Entrée pour quitter...")
