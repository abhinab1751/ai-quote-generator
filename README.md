# 🧠 Oracle — AI Quote Generator

> An end-to-end AI-powered quote generation web app built with a custom LSTM language model, Flask REST API, and a clean HTML/CSS/JS frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Overview

Oracle is an AI quote generator that uses a custom-trained LSTM (Long Short-Term Memory) neural network to generate original, human-like quotes based on a topic or seed phrase provided by the user. The model was trained on a dataset of real quotes and learns the patterns, rhythm, and structure of meaningful language.

Unlike retrieval-based quote apps that just fetch quotes from a database, Oracle **generates entirely new quotes** that have never existed before — every output is unique.

---

## Features

- 🎯 **Topic-based generation** — Enter any seed phrase or topic and get AI-generated quotes around it
- 🔢 **Quantity control** — Generate 1 to 5 quotes in a single request
- 🌡️ **Temperature sampling** — Internally uses temperature-based sampling for varied, non-repetitive output
- ⚡ **Fast inference** — Model loads once at startup, all requests are served instantly
- 🌐 **REST API** — Clean Flask backend, easily extendable
- 🎨 **Polished UI** — Editorial-style frontend with smooth animations and one-click copy

---

## Project Structure

```
ai-quote-generator/
│
├── backend/
│   ├── app.py              # Flask REST API server
│   ├── quote_model.h5      # Trained LSTM model weights
│   ├── tokenizer.pkl       # Fitted tokenizer (must match training)
│   ├── config.pkl          # Training config (max_sequence_len etc.)
│   ├── Procfile            # Render deployment entry point
│   └── requirements.txt    # Python dependencies
│
├── model/
│   ├── train.py            # Model training script
│   ├── generate.py         # Standalone quote generation script
│   ├── preprocess.py       # Data cleaning and preprocessing
│   └── check_data.py       # Data validation utility
│
├── data/
│   ├── quotes.json         # Raw quotes dataset
│   ├── quotes.txt          # Plain text version
│   └── quotes_clean.txt    # Cleaned and filtered quotes
│
├── frontend/
│   └── index.html          # Single-file frontend (HTML + CSS + JS)
│
└── README.md
```

---

## How It Works

```
User types topic → Frontend sends POST request → Flask API receives it
       ↓
Seed text is tokenized using the saved tokenizer
       ↓
LSTM model predicts next word (with temperature sampling)
       ↓
Process repeats for 20 words
       ↓
Generated quote is returned as JSON → Displayed on screen
```

### Temperature Sampling

Instead of always picking the single most likely next word (which produces repetitive, boring output), the model uses **temperature sampling**:

- **High temperature (1.2)** → more creative, unpredictable output
- **Medium temperature (0.8)** → balanced, natural-sounding quotes
- **Low temperature (0.5)** → conservative, coherent but less varied

The backend currently uses `temperature=0.8` by default for the best balance of quality and variety.

---

## Model Architecture

```
Input (seed tokens)
      ↓
Embedding Layer     — 64 dimensions, maps token IDs to dense vectors
      ↓
LSTM Layer          — 128 units, learns sequential patterns in language
      ↓
Dropout Layer       — 20% dropout to prevent overfitting
      ↓
Dense + Softmax     — outputs probability over entire vocabulary
      ↓
Temperature Sample  — picks next word based on probability distribution
```

### Training Details

| Parameter | Value |
|---|---|
| Dataset | quotes.json (~8,000 quotes) |
| Vocabulary size | ~6,000–8,000 words (rare words filtered) |
| Max sequence length | 25 tokens |
| Embedding dimensions | 64 |
| LSTM units | 128 |
| Dropout | 0.2 |
| Optimizer | Adam (lr=0.005) |
| Batch size | 512 |
| Max epochs | 30 (early stopping, patience=5) |
| Training time | ~30–45 minutes on CPU |

**Early stopping** halts training automatically once the loss stops improving, preventing overfitting and saving time. **ReduceLROnPlateau** halves the learning rate when the model plateaus, helping it converge more precisely.

---

## Installation

### Prerequisites

- Python 3.10+
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOURUSERNAME/ai-quote-generator.git
cd ai-quote-generator
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

---

## Usage

### Train the model (optional — pretrained model included)

```bash
cd model
python train.py
```

This will generate `quote_model.h5`, `tokenizer.pkl`, and `config.pkl` inside the `model/` folder. Copy these into `backend/` before running the server.

### Run the backend server

```bash
cd backend
python app.py
```

Server starts at `http://127.0.0.1:5000`

### Open the frontend

Simply open `frontend/index.html` in your browser. It connects to the Flask backend automatically.

### Test quotes from terminal (standalone)

```bash
cd model
python generate.py
```

---

## API Reference

### `POST /generate`

Generates one or more quotes based on a seed topic.

**Request body:**
```json
{
  "topic": "life is",
  "count": 3
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | string | No | Seed phrase for generation. Defaults to `"life"` |
| `count` | integer | No | Number of quotes to generate (1–5). Defaults to `1` |

**Response:**
```json
{
  "quotes": [
    "life is a journey of moments we choose to remember",
    "life is the art of learning from what we cannot change",
    "life is not measured by time but by the depth of our presence"
  ],
  "topic": "life is"
}
```

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

## Deployment

This app is deployed with:
- **Backend** → [Render](https://render.com) (free tier, Flask + gunicorn)
- **Frontend** → [Vercel](https://vercel.com) (free tier, static hosting)

### Deploy backend to Render

1. Push the `backend/` folder contents to a GitHub repo
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo and set:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Deploy — your API will be live at `https://your-app.onrender.com`

### Deploy frontend to Vercel

1. Update the API URL in `frontend/index.html`:
```js
const API = "https://your-app.onrender.com/generate";
```
2. Go to [vercel.com](https://vercel.com) → **New Project** → import repo
3. Set **Root Directory** to `frontend`
4. Deploy

> ⚠️ **Note:** Render free tier spins down after 15 minutes of inactivity. The first request after idle may take ~30 seconds to wake up.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | TensorFlow / Keras |
| Model Type | LSTM (Long Short-Term Memory) |
| Backend | Flask + Flask-CORS |
| Server | Gunicorn |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Fonts | Playfair Display, Space Mono |
| Backend Hosting | Render |
| Frontend Hosting | Vercel |
| Language | Python 3.10+ |

---

## License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

<p align="center">Built with 🤖 and ☕ — Oracle AI Quote Generator</p>