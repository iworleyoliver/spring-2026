"""
Train a Perceptron on a small synthetic dataset and print accuracy.
Run: python3 train.py
"""
import random
import os
from model import Perceptron

random.seed(42)


def make_linearly_separable(n=200):
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)
        # decision boundary: x1 + x2 > 0 -> class 1
        label = 1 if x1 + x2 + random.uniform(-0.5, 0.5) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def train_and_save():
    X, y = make_linearly_separable(400)
    # simple shuffle and split
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    p = Perceptron(lr=0.01, n_iter=20)
    p.fit(X_train, y_train)
    train_acc = p.score(X_train, y_train)
    test_acc = p.score(X_test, y_test)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "perceptron.pkl")
    p.save(model_path)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_and_save()
