##import libraries
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input,Embedding,LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Step 2: Load dataset

from datasets import load_dataset

dataset = load_dataset("asoria/english-quotes-text")

print(dataset["train"][0])
print(dataset["train"].column_names)


##Step 3: cleaning data
import re

raw_lines = dataset["train"]["text"]

clean_lines = []

for line in raw_lines:

    line = line.split("Author")[0] ## removes author info
    line = line.replace("Quote:", "") ##removes the word quotw
    line = line.lower() ## converts to lowercase
    line = re.sub(r"[^a-zA-Z\s]", "", line) ## removes punctuation and special characters
    line = " ".join(line.split()) ## removes extra spaces
    if len(line.split()) >= 2:  ## keeps lines with at least 2 words
        clean_lines.append(line)

print("Total cleaned lines:", len(clean_lines))
print(clean_lines[:5])

#After cleaning your dataset, remove exact duplicates:
clean_lines = list(dict.fromkeys(clean_lines))

train_lines, val_lines = train_test_split(
    clean_lines,
    test_size=0.1,
    random_state=42
)


# Step 4: Tokenization

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_lines)

total_words = len(tokenizer.word_index) + 1

print("Vocabulary size:", total_words)
print("First 20 words:", list(tokenizer.word_index.items())[:20])


 ##testing tokenizarion
ex = clean_lines[0]
print("Example line:",ex)
encoded = tokenizer.texts_to_sequences([ex])[0]
print("Encoded line:",encoded)

# Step 5: Create next-word training sequences
def create_sequences(lines, tokenizer):

    sequences = []

    for line in lines:

        tokens = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(tokens)):
            sequences.append(tokens[:i + 1])

    return sequences

train_sequences = create_sequences(train_lines, tokenizer)
val_sequences = create_sequences(val_lines, tokenizer)


#step 6: Padding the input sequences
# max_sequence_len = max(len(seq) for seq in input_sequences)
max_sequence_len = 30

train_sequences = np.array(
    pad_sequences(
        train_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )
)

val_sequences = np.array(
    pad_sequences(
        val_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )
)



## Step 7 Separate input X and Target y

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]

X_val = val_sequences[:, :-1]
y_val = val_sequences[:, -1]


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

## Step 8: Build the LSTM Model

## Model flow : INput -> Embedding -> LSTM -> Dropout -> Dense Softmax
## Embedding Turn each word ID into a learned vector
lstm_model= Sequential([
    Input(shape=(max_sequence_len -1,)),
    Embedding(input_dim = total_words, output_dim = 64,mask_zero=True
),
    LSTM(units = 128),
    Dropout(rate = 0.2),
    Dense(units = total_words,activation = "softmax")
])



lstm_model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]

)

lstm_model.summary()

## Train the LSTM Model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

EPOCHS = 20

BATCH_SIZE = 120
lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)
## Step to Build the GRU Model

gru_model = Sequential([
    Input(shape=(max_sequence_len -1,)),
    Embedding(input_dim = total_words, output_dim = 64, mask_zero=True),
    GRU(units = 128),
    Dropout(rate = 0.2),
    Dense(units = total_words, activation = "softmax")
]
)

gru_model.compile(
 loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"])

gru_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[gru_early_stopping],
    verbose=1
)


### compare validation accuracy of LSTM and GRU
plt.figure(figsize = (9,5))
plt.plot(lstm_history.history["val_accuracy"],label = "LSTM Validation Accuracy")
plt.plot(gru_history.history["val_accuracy"],label = "GRU Validation Accuracy")

plt.title("LSTM vs GRU - Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.show()


## compare validation loss of LSTM and GRU
plt.figure(figsize = (9,5))
plt.plot(lstm_history.history["val_loss"],label = "LSTM Validation loss")
plt.plot(gru_history.history["val_loss"],label = "GRU Validation loss")

plt.title("LSTM vs GRU - Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.show()


# Step 13: Predict Next Words

def predict_next_words(seed_text, model, top_k=5):

    # Clean user input
    seed_text = seed_text.lower()
    seed_text = re.sub(r"[^a-zA-Z\s]", "", seed_text)
    seed_text = " ".join(seed_text.split())

    if not seed_text:
        return []

    # Convert text to token IDs
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre",
        truncating="pre"
    )

    # Predict next-word probabilities
    predicted = model.predict(token_list, verbose=0)[0]

    # Sort words by probability
    top_indices = np.argsort(predicted)[::-1]

    predictions = []

    for index in top_indices:

        word = tokenizer.index_word.get(int(index), "")

        # Skip invalid tokens and OOV
        if not word or word.lower() == "<oov>":
            continue

        predictions.append(
            (word, float(predicted[index]))
        )

        if len(predictions) == top_k:
            break

    return predictions


# Step 14: Test Next-Word Prediction

seed_text = "be yourself"

print("\nLSTM Predictions:")

lstm_predictions = predict_next_words(
    seed_text,
    lstm_model,
    top_k=5
)

for word, probability in lstm_predictions:
    print(word, ":", round(probability, 4))


print("\nGRU Predictions:")

gru_predictions = predict_next_words(
    seed_text,
    gru_model,
    top_k=5
)

for word, probability in gru_predictions:
    print(word, ":", round(probability, 4))



# Step 15: Generate a short continuation

# Predict -> Append -> Predict again -> Append again

def generate_words(seed_text, model, number_of_words=8):

    generated_text = seed_text.strip()

    for _ in range(number_of_words):

        predictions = predict_next_words(
            generated_text,
            model,
            top_k=1
        )

        if not predictions:
            break

        generated_text += " " + predictions[0][0]

    return generated_text


print("LSTM:", generate_words("be yourself", lstm_model, 8))
print("GRU:", generate_words("be yourself", gru_model, 8))

lstm_model.save("lstm_model.keras")
gru_model.save("gru_model.keras")

print("Both models saved successfully!")

# Save tokenizer
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Save sequence length
with open("config.json", "w") as file:
    json.dump(
        {"max_sequence_len": int(max_sequence_len)},
        file
    )

print("Tokenizer and configuration saved!")