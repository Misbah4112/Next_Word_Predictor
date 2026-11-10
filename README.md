# Next Word Predictor | LSTM & GRU

A deep learning project that predicts the next word in a sentence using **LSTM and GRU neural networks**. Built to explore Recurrent Neural Networks (RNNs), natural language processing, and text generation.

## Tech Stack
- **Python**
- **TensorFlow / Keras** — LSTM and GRU models
- **Streamlit** — Interactive web application
- **NumPy & Pandas** — Data processing
- **Plotly** — Interactive visualizations

## Features
- **Next Word Prediction:** Predicts the most likely next words with confidence scores.
- **Text Generation:** Generates sentence continuations using greedy decoding.
- **Model Comparison:** Compares LSTM and GRU predictions side by side.
- **Interactive Dashboard:** Visualizes prediction probabilities.

## Dataset
Trained on the **English Quotes Dataset** (`asoria/english-quotes-text`) from Hugging Face.

## How It Works
1. Clean and tokenize text data.
2. Generate input sequences for next-word prediction.
3. Train separate LSTM and GRU models.
4. Predict the next word using softmax probabilities.
5. Generate text by repeatedly selecting the highest-probability word.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Scope
An educational deep learning project focused on understanding sequence modeling, next-word prediction, and the differences between LSTM and GRU architectures.
