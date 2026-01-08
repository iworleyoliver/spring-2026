"""
Simple Perceptron implementation with no external dependencies.
Provides fit, predict, score, save, and load.
"""
import pickle
from typing import List

class Perceptron:
    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights: List[float] | None = None
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            raise ValueError("X must be non-empty")
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                linear = sum(w * xij for w, xij in zip(self.weights, xi)) + self.bias
                y_pred = 1 if linear >= 0 else 0
                error = yi - y_pred
                if error != 0:
                    for j in range(n_features):
                        self.weights[j] += self.lr * error * xi[j]
                    self.bias += self.lr * error

    def predict(self, X: List[List[float]]) -> List[int]:
        if self.weights is None:
            raise ValueError("Model is not trained yet")
        preds: List[int] = []
        for xi in X:
            linear = sum(w * xij for w, xij in zip(self.weights, xi)) + self.bias
            preds.append(1 if linear >= 0 else 0)
        return preds

    def score(self, X: List[List[float]], y: List[int]) -> float:
        preds = self.predict(X)
        correct = sum(int(p == t) for p, t in zip(preds, y))
        return correct / len(y) if y else 0.0

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "bias": self.bias, "lr": self.lr, "n_iter": self.n_iter}, f)

    @staticmethod
    def load(path: str) -> "Perceptron":
        with open(path, "rb") as f:
            data = pickle.load(f)
        p = Perceptron(lr=data.get("lr", 0.1), n_iter=data.get("n_iter", 1000))
        p.weights = data.get("weights")
        p.bias = data.get("bias", 0.0)
        return p
