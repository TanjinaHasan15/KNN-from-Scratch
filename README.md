# KNN from Scratch: Flower and News Classification with Custom Evaluation Metrics

##  Project Overview

This project demonstrates how to implement the **K-Nearest Neighbors (KNN)** algorithm from scratch in Python. It includes:

- Manual implementation of KNN
- Custom evaluation metrics: accuracy, confusion matrix, precision, recall, and F1-score
- Classification of:
  - The **Iris flower dataset**
  - A small **custom news dataset** (Politics vs Sports)
- Comparison between custom KNN and **Scikit-learn's KNN**
- Experimentation with K value and train-test split ratio

---

##  Concepts Covered

- Lazy vs Eager Learning
- Euclidean Distance
- TF-IDF Vectorization for text classification
- Custom implementation of evaluation metrics
- Model comparison between custom and sklearn KNN

---

##  Output
```text
Custom KNN on Iris Dataset:
Accuracy: 0.96
Confusion Matrix:
 [[19  0  0]
  [ 0 12  1]
  [ 0  0 13]]
Precision: [1.0, 0.92, 0.93]
Recall:    [1.0, 0.92, 1.0]
F1-Score:  [1.0, 0.92, 0.96]

Custom KNN on News Dataset:
Accuracy: 0.33
Confusion Matrix:
 [[1 0]
  [2 0]]
Precision: [0.33, 0.0]
Recall:    [1.0, 0.0]
F1-Score:  [0.5, 0.0]

Scikit-learn KNN on News Dataset:
Accuracy: 0.33
Confusion Matrix:
 [[1 0]
  [2 0]]
Precision: [0.33, 0.0]
Recall:    [1.0, 0.0]
F1-Score:  [0.5, 0.0]
```



