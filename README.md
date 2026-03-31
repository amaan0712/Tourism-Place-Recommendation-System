# 🗺️ Tourism Place Recommendation System

A deep learning-based collaborative filtering system that recommends tourism places to users based on their past ratings. The model uses a **Neural Collaborative Filtering (NCF)** architecture — combining Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP) — to learn rich user-place interaction patterns.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Output](#output)

---

## Overview

This project builds a personalized tourism recommendation engine trained on user ratings of tourist destinations. Given a user, the model predicts how much they would enjoy places they haven't visited yet, and returns the top-N recommendations.

**Key highlights:**
- Neural Collaborative Filtering (NCF) combining GMF + MLP branches
- Normalized ratings with train/validation/test splits
- Early stopping, learning rate scheduling, and model checkpointing
- Interactive visualizations using Plotly and Matplotlib

---

## Dataset

The project uses three CSV files placed inside a Google Drive folder (`Dataset/`):

| File | Description |
|---|---|
| `tourism_rating.csv` | User–place rating interactions (`User_Id`, `Place_Id`, `Place_Ratings`) |
| `tourism_with_id.csv` | Place metadata (`Place_Id`, `Place_Name`, `Category`, `City`, etc.) |
| `user.csv` | User metadata (`User_Id`, age, location, etc.) |

> **Note:** You must upload these files to your Google Drive before running the notebook. The expected path is `MyDrive/Dataset/`.

---

## Project Structure

```
tourism-recommendation/
│
├── tourism_recommendation.ipynb   # Main Colab notebook
├── README.md                      # This file
└── Dataset/                       # (on Google Drive)
    ├── tourism_rating.csv
    ├── tourism_with_id.csv
    └── user.csv
```

---

## Requirements

All dependencies are installed automatically at the top of the notebook:

```
pandas
numpy
scikit-learn
matplotlib
tensorflow
plotly
```

**Environment:** Google Colab (recommended) — GPU runtime preferred for faster training.

---

## Setup & Installation

### Step 1 — Open in Google Colab

Upload or open `tourism_recommendation.ipynb` in [Google Colab](https://colab.research.google.com/).

### Step 2 — Enable GPU (optional but recommended)

Go to **Runtime → Change runtime type → Hardware accelerator → GPU**.

### Step 3 — Upload the Dataset to Google Drive

Place the three CSV files in a folder called `Dataset` inside your Google Drive root (`My Drive/Dataset/`).

### Step 4 — Mount Google Drive

The notebook will prompt you to authorize Google Drive access when you run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the authorization link and paste the code when prompted.

### Step 5 — Install Dependencies

The first cell installs all required packages:

```python
!pip install -q pandas numpy scikit-learn matplotlib tensorflow plotly
```

---

## How to Run

Once setup is complete, run all cells in order from top to bottom (**Runtime → Run all**).

The notebook will:

1. **Load & merge** the three datasets into a single interaction dataframe
2. **Visualize** rating distributions, user/place activity, and category breakdowns
3. **Preprocess** data — encode users and places, normalize ratings, split into train/val/test sets
4. **Build** the NCF model with GMF + MLP branches
5. **Train** the model with early stopping and learning rate scheduling
6. **Evaluate** performance on the held-out test set (RMSE, MAE)
7. **Generate** top-N recommendations for a given user

---

## Model Architecture

The model is a **Neural Collaborative Filtering (NCF)** network with two parallel branches that are fused before the final prediction:

```
User Input ──► GMF User Embedding ─┐
                                    ├─► Element-wise Multiply ──► GMF Output ─┐
Place Input ──► GMF Place Embedding ┘                                         │
                                                                               ├─► Concatenate ──► sigmoid ──► Rating
User Input ──► MLP User Embedding ─┐                                          │
                                    ├─► Concatenate ──► Dense(256) ──►        │
Place Input ──► MLP Place Embedding ┘   BatchNorm + Dropout                   │
                                    ──► Dense(128) ──► Dense(64) ──►          │
                                        Dense(32) ──► MLP Output ─────────────┘
```

**Key components:**

| Component | Detail |
|---|---|
| GMF branch | Embedding dim = `embeddingDim // 2`, element-wise product |
| MLP branch | Embedding dim = `embeddingDim`, Dense layers: 256 → 128 → 64 → 32 |
| Regularization | L2 on embeddings, BatchNorm + Dropout (0.3, 0.25, 0.2) |
| Output activation | Sigmoid (predicts normalized rating in [0, 1]) |
| Loss | Mean Squared Error (MSE) |
| Optimizer | Adam |

---

## Configuration

All key hyperparameters are defined as variables near the top of the notebook and can be changed without touching the model code:

```python
epochs         = 40       # Maximum training epochs
embeddingDim   = 64       # Embedding size (GMF uses half)
batchSize      = 256      # Training batch size
learningRate   = 0.001    # Initial Adam learning rate
validationSplit= 0.15     # Fraction of data used for validation
testSplit      = 0.10     # Fraction of data held out for testing
topN           = 10       # Number of recommendations to generate
```

---

## Output

After training completes, the notebook produces:

- **Training curves** — loss and MAE across epochs for train and validation sets
- **Test metrics** — RMSE and MAE on the held-out test set
- **Top-N recommendations** — a ranked list of places for a given user, including place name, category, and city
- **Interactive charts** — Plotly bar and histogram visualizations of the dataset

---

## Notes

- The random seed is fixed (`tf.random.set_seed(42)`, `np.random.seed(42)`) for reproducibility.
- The model uses `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks to prevent overfitting and save the best weights automatically.
- Ratings are min-max normalized to [0, 1] before training and de-normalized when displaying recommendations.
