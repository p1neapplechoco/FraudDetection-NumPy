import numpy as np


class DecisionTree:
    """Simple Decision Tree for Random Forest."""

    def __init__(
        self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        """Calculate Gini impurity."""
        _, counts = np.unique(y, return_counts=True)
        n = len(y)
        probabilities = counts / n
        return 1 - np.einsum("i,i->", probabilities, probabilities)

    def _split(self, X, y, feature_idx, threshold):
        """Split data based on feature and threshold."""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask

    def _best_split(self, X, y, feature_indices):
        """Find the best split."""
        best_gini = float("inf")
        best_feature = None
        best_threshold = None
        n = len(y)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            thresholds = np.unique(sorted_values)

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                n_left = np.sum(left_mask)
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[~left_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (
            (depth >= self.max_depth if self.max_depth is not None else False)
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            return {"leaf": True, "value": np.bincount(y.astype(int)).argmax()}

        # Select random features
        if self.max_features is not None:
            feature_indices = np.random.choice(
                n_features, self.max_features, replace=False
            )
        else:
            feature_indices = np.arange(n_features)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        if best_feature is None:
            return {"leaf": True, "value": np.bincount(y.astype(int)).argmax()}

        # Split data
        left_mask, right_mask = self._split(X, y, best_feature, best_threshold)

        # Build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def fit(self, X, y):
        """Fit the decision tree."""
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        """Predict a single sample."""
        if node["leaf"]:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])

    def predict(self, X):
        """Predict multiple samples."""
        n_samples = X.shape[0]

        predictions = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            predictions[i] = self._predict_sample(X[i], self.tree)

        return predictions


class RandomForest:
    def __init__(
        self,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """Fit the random forest."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Determine max_features
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features

        self.trees = []

        for _ in range(self.n_trees):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """Predict using majority voting."""
        n_samples = X.shape[0]
        predictions = np.empty((len(self.trees), n_samples), dtype=int)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        result = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            result[i] = np.bincount(predictions[:, i]).argmax()
        return result

    def predict_proba(self, X):
        """Predict class probabilities."""
        n_samples = X.shape[0]

        predictions = np.empty((len(self.trees), n_samples), dtype=int)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        n_classes = len(np.unique(predictions))

        proba = np.zeros((n_samples, n_classes))
        for class_idx in range(n_classes):
            proba[:, class_idx] = np.sum(predictions == class_idx, axis=0)

        proba /= self.n_trees
        return proba
