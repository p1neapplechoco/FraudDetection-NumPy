import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_predicted):
        n = len(y)
        loss = -(1 / n) * np.sum(
            y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted)
        )
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.__sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss_value = self.loss(y, y_predicted)
            self.loss_history.append(loss_value)

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            raise Exception("Model is not trained yet. Please call 'fit' first.")
        linear_model = np.dot(X, self.weights) + self.bias
        return self.__sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
