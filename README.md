# Neural Network from Scratch — Lab 2

## Overview
Implementation of a fully connected neural network using NumPy, trained on the scikit-learn Digits dataset.

## Architecture
- Input layer: 64 features
- Two hidden layers with ReLU activation
- Output layer: 10 classes with softmax
- Loss function: Cross-entropy
- Weight initialization: Normal distribution with mean 0 and standard deviation 0.01
- Bias initialization: Zeros

## Experiments
Hidden layer widths: 16 and 64

Learning rates: 0.001, 0.01, and 0.1

Training: 50 epochs, mini-batch size 32

## Validation
- Numerical gradient checking
- Training and test accuracy
- Forward and backward pass timing
- Loss curves saved in `results/loss_curves.png`

## Technologies
- Python
- NumPy
- scikit-learn
- Matplotlib
