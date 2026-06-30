# Deep Learning ANN Pipeline for Customer Churn Prediction

This repository implements an end-to-end deep learning binary classification pipeline designed to predict customer churn risks. By ingestive modeling of demographic attributes, account state configurations, and behavioral transaction metrics, the system trains a multi-layer Artificial Neural Network (ANN) to map non-linear correlations and flag high-risk customer profiles prior to contract termination.

---

## Overview

- Core Framework: TensorFlow / Keras
- Model Architecture: Multi-Layer Perceptron (MLP) fully-connected ANN
- Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix
- Preprocessing Stack: Scikit-Learn, Pandas, NumPy

---

## Features
- Structured deep learning pipeline utilizing fully connected dense hidden layers with ReLU activation matrices
- Normalized binary classification output powered by a targeted Sigmoid threshold neuron
- End-to-end data preprocessing layer covering high-performance feature scaling, normalization, and categorical encoding
- Robust model evaluation engines generating actionable confusion matrices, precision curves, and tracking metrics
- Modular project architecture partitioning raw preprocessing workflows from training execution runs

---

## Tech Stack

| Component | Tool / Service |
| :--- | :--- |
| Core Language | Python |
| Deep Learning | TensorFlow, Keras |
| Preprocessing & Metrics | Scikit-Learn, Pandas, NumPy |
| Data Visualization | Matplotlib |

---

## Network Architecture & Pipeline Workflow

1. **Feature Engineering & Preprocessing**
   Transforms raw heterogeneous fields (demographics, transactional history, account balances) into standardized vector tensors using standard matrix scaling and encoding techniques.

2. **Input Layer Layering**
   Accepts the compiled dense numerical feature arrays and passes them systematically to downstream computational units.

3. **Hidden Transformation Layers**
   Utilizes fully connected dense layers embedded with Rectified Linear Unit (ReLU) activations to map intricate non-linear relationships across user profiles.

4. **Output Binary Classification**
   Terminates at a single-neuron Sigmoid layer, computing a probabilistic churn index score bounded explicitly between 0 and 1.
