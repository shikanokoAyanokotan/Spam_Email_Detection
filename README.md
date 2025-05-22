
# Spam Email Detection Using LSTM

This project demonstrates how to build a spam email detector using natural language processing and a deep learning model (LSTM) with PyTorch.

## 1. Data Loading

We begin by loading a labeled dataset of emails, which includes both spam and non-spam messages. The data is read using pandas and explored for initial insights.

## 2. Preprocessing Data

### 2.1 Clean Text and Tokenize

Text data is cleaned by:
- Removing HTML tags
- Removing non-alphabetic characters
- Lowercasing all text
- Tokenizing each email

### 2.2 Encoding

The tokenized words are converted into integer sequences using a vocabulary dictionary. Each unique word is mapped to an integer ID.

### 2.3 Padding Sequence

Since LSTM requires inputs of the same length, all sequences are padded to the same maximum length using zeros.

## 3. Processing Data

### 3.1 Splitting Data

The dataset is split into training and testing sets to evaluate model performance properly.

### 3.2 Custom PyTorch Dataset and DataLoader

A custom PyTorch `Dataset` class is created to feed the model. `DataLoader` is used to handle batching and shuffling during training.

### 3.3 Defining the LSTM

An LSTM neural network is defined using PyTorch’s `nn.Module`. It includes:
- An embedding layer
- One or more LSTM layers
- Fully connected layers
- Sigmoid activation for binary classification

## 4. Training

The model is trained using binary cross-entropy loss and an optimizer such as Adam. The loss and accuracy are tracked across epochs.

## 5. Evaluation

After training, the model is evaluated on the test set using accuracy, precision, recall, and F1-score to assess its spam detection capability.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- torch
- tqdm

---
