# Chess Move-Based Win Prediction using RNN-GRU

This project uses a Recurrent Neural Network with Gated Recurrent Units (GRU) to predict whether white or black is more likely to win a chess game, based only on the sequence of moves played.

## Overview

The model is trained on historical chess game data from Lichess.org. It takes in the sequence of moves from a game and outputs a binary classification: white win or black win. This can be applied to evaluate openings, identify advantageous patterns, or enhance real-time game commentary.

## Technologies Used

- Python
- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn
- Jupyter Notebook

## Dataset

- Source: Lichess.org
- Size: ~20,000 games
- Features used: move sequence and winner
- Preprocessing steps:
  - Tokenize each move
  - Pad sequences to uniform length
  - Encode labels for binary classification

## Model Architecture

- Type: Recurrent Neural Network
- Variant: GRU (Gated Recurrent Unit)
- Activation: Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Training: 3 epochs with 20% validation split

## Training Process

1. Load and clean the dataset
2. Extract all possible unique moves
3. Convert move sequences to integer tokens
4. Pad sequences to the length of the longest game
5. Split into training and testing sets (70/30)
6. Train GRU-based model on training set
7. Evaluate accuracy on validation and test sets

## Results

| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 78%                 |
| 2     | 82%                 |
| 3     | 74%                 |
| Average | 77%              |

The model performs well when given full game sequences. With partial sequences, performance decreases but still maintains useful predictive capability.

## Limitations

- Data contains noise due to resignations and anomalies
- Dataset is relatively small given the vast number of possible chess positions
- Model only considers move sequence, not the board state or piece positions

## Future Work

- Train on a much larger dataset
- Include board state features or evaluations
- Explore more advanced architectures (e.g., LSTM, Transformer)
- Build a real-time prediction tool for live games


