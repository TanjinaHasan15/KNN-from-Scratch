# KNN from Scratch: Flower and News Classification with Custom Evaluation Metrics

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix as sk_cm, accuracy_score as sk_acc

# --- Custom KNN Class ---
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        return [self._predict_one(x) for x in X_test]

    def _predict_one(self, x):
        distances = [self._euclidean(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

# --- Custom Evaluation Metrics ---
def accuracy_score_custom(y_true, y_pred):
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

def confusion_matrix_custom(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        matrix[label_to_index[t]][label_to_index[p]] += 1
    return matrix

def precision_recall_f1_custom(y_true, y_pred, labels=None):
    cm = confusion_matrix_custom(y_true, y_pred, labels)
    precision, recall, f1 = [], [], []
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1_score = 2 * prec * rec / (prec + rec) if prec + rec else 0
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    return precision, recall, f1

# ---  Iris Dataset ---
def run_iris():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNN(k=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score_custom(y_test, y_pred)
    cm = confusion_matrix_custom(y_test, y_pred)
    prec, rec, f1 = precision_recall_f1_custom(y_test, y_pred)

    print("\nCustom KNN on Iris Dataset:")
    print(f"Accuracy: {round(acc, 2)}")
    print("Confusion Matrix:\n", cm)
    print("Precision:", [round(p, 2) for p in prec])
    print("Recall:   ", [round(r, 2) for r in rec])
    print("F1-Score: ", [round(f, 2) for f in f1])

# --- Custom News Dataset ---
def run_news_custom():
    news_data = [
        ("The government passed a new economic bill in the parliament today.", "politics"),
        ("The local team won the championship after a thrilling final match.", "sports"),
        ("Prime Minister addressed the nation regarding the budget changes.", "politics"),
        ("Star striker scored a hat-trick in the international friendly.", "sports"),
        ("Election campaigns are heating up across the major states.", "politics"),
        ("The national cricket team starts their tour next week.", "sports"),
        ("A controversial bill sparked debates in the Senate.", "politics"),
        ("The tennis tournament drew record attendance this year.", "sports"),
        ("Foreign policies are being restructured by the new cabinet.", "politics"),
        ("Fans celebrated wildly after their team’s last-minute win.", "sports")
    ]
    texts, labels = zip(*news_data)
    labels_num = [0 if label == "politics" else 1 for label in labels]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels_num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNN(k=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score_custom(y_test, y_pred)
    cm = confusion_matrix_custom(y_test, y_pred)
    prec, rec, f1 = precision_recall_f1_custom(y_test, y_pred)

    print("\nCustom KNN on News Dataset:")
    print(f"Accuracy: {round(acc, 2)}")
    print("Confusion Matrix:\n", cm)
    print("Precision:", [round(p, 2) for p in prec])
    print("Recall:   ", [round(r, 2) for r in rec])
    print("F1-Score: ", [round(f, 2) for f in f1])

    # ---  KNN on News Dataset ---
def run_news_sklearn():
    news_data = [
        ("The government passed a new economic bill in the parliament today.", "politics"),
        ("The local team won the championship after a thrilling final match.", "sports"),
        ("Prime Minister addressed the nation regarding the budget changes.", "politics"),
        ("Star striker scored a hat-trick in the international friendly.", "sports"),
        ("Election campaigns are heating up across the major states.", "politics"),
        ("The national cricket team starts their tour next week.", "sports"),
        ("A controversial bill sparked debates in the Senate.", "politics"),
        ("The tennis tournament drew record attendance this year.", "sports"),
        ("Foreign policies are being restructured by the new cabinet.", "politics"),
        ("Fans celebrated wildly after their team’s last-minute win.", "sports")
    ]
    texts, labels = zip(*news_data)
    labels_num = [0 if label == "politics" else 1 for label in labels]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels_num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = sk_acc(y_test, y_pred)
    cm = sk_cm(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    print("\nScikit-learn KNN on News Dataset:")
    print(f"Accuracy: {round(acc, 2)}")
    print("Confusion Matrix:\n", cm)
    print("Precision:", [round(p, 2) for p in prec])
    print("Recall:   ", [round(r, 2) for r in rec])
    print("F1-Score: ", [round(f, 2) for f in f1])