import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def __sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def loss(self, y, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        return -np.mean(np.where(y == 1, np.log(y_predicted), np.log(1 - y_predicted)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        inv_n = 1.0 / n_samples

        for _ in range(self.n_iterations):
            linear_model = np.einsum("ij,j->i", X, self.weights) + self.bias
            y_predicted = self.__sigmoid(linear_model)

            error = y_predicted - y
            dw = np.einsum("ji,j->i", X, error) * inv_n
            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss_value = self.loss(y, y_predicted)
            self.loss_history.append(loss_value)

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            raise Exception("Model is not trained yet. Please call 'fit' first.")

        linear_model = np.einsum("ij,j->i", X, self.weights) + self.bias
        return self.__sigmoid(linear_model)

    def predict(self, X, threshold=0.5):

        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
