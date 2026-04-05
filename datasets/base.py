import numpy as np


class Dataset:
    def __init__(
        self,
        X,
        y=None,
        task="unsupervised",
        name=None,
        feature_names=None,
        target_name=None,
    ):
        """
        Generic dataset container.

        Parameters:
        - X: Features (numpy array or array-like)
        - y: Labels (optional)
        - task: "classification", "regression", "unsupervised"
        - name: Dataset name
        - feature_names: List of feature names
        - target_name: Name of target column
        """
        self.X = np.array(X)
        self.y = np.array(y) if y is not None else None
        self.task = task
        self.name = name or "dataset"
        self.feature_names = feature_names
        self.target_name = target_name

        self._validate()

    # ------------------------
    # Internal validation
    # ------------------------
    def _validate(self):
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")

        if self.task not in ["classification", "regression", "unsupervised"]:
            raise ValueError(f"Invalid task type: {self.task}")

    # ------------------------
    # Basic properties
    # ------------------------
    def __len__(self):
        return len(self.X)

    @property
    def shape(self):
        return self.X.shape

    @property
    def n_samples(self):
        return self.X.shape[0]

    @property
    def n_features(self):
        return self.X.shape[1]

    # ------------------------
    # Copying
    # ------------------------
    def copy(self):
        return Dataset(
            X=self.X.copy(),
            y=self.y.copy() if self.y is not None else None,
            task=self.task,
            name=self.name,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

    # ------------------------
    # Train/test split
    # ------------------------
    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None):
        rng = np.random.default_rng(random_state)

        indices = np.arange(self.n_samples)

        if shuffle:
            rng.shuffle(indices)

        split = int(self.n_samples * (1 - test_size))
        train_idx, test_idx = indices[:split], indices[split:]

        return (
            self._subset(train_idx),
            self._subset(test_idx),
        )

    def _subset(self, indices):
        X = self.X[indices]
        y = self.y[indices] if self.y is not None else None

        return Dataset(
            X=X,
            y=y,
            task=self.task,
            name=self.name,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

    # ------------------------
    # Transformations
    # ------------------------
    def normalize(self):
        """Standard score normalization (z-score)"""
        X = self.X.copy()
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        std[std == 0] = 1  # prevent division by zero

        X = (X - mean) / std

        return Dataset(
            X=X,
            y=self.y,
            task=self.task,
            name=self.name,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

    def minmax_scale(self):
        """Scale features to [0, 1]"""
        X = self.X.copy()
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)

        denom = max_val - min_val
        denom[denom == 0] = 1

        X = (X - min_val) / denom

        return Dataset(
            X=X,
            y=self.y,
            task=self.task,
            name=self.name,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

    def shuffle(self, random_state=None):
        rng = np.random.default_rng(random_state)
        indices = np.arange(self.n_samples)
        rng.shuffle(indices)
        return self._subset(indices)

    # ------------------------
    # Data inspection
    # ------------------------
    def summary(self):
        print(f"Dataset: {self.name}")
        print(f"Task: {self.task}")
        print(f"Samples: {self.n_samples}")
        print(f"Features: {self.n_features}")
        print(f"Feature names: {self.feature_names if self.feature_names is not None else 'N/A'}")

        if self.y is not None:
            print(f"Target available: Yes")
            print(f"Target name: {self.target_name if self.target_name is not None else 'N/A'}")

            if self.task == "classification":
                unique, counts = np.unique(self.y, return_counts=True)
                print("Class distribution:")
                for u, c in zip(unique, counts):
                    print(f"  {u}: {c}")
        else:
            print("Target available: No")

    # ------------------------
    # Feature/target handling
    # ------------------------
    def select_features(self, indices):
        X = self.X[:, indices]

        feature_names = None
        if self.feature_names is not None:
            feature_names = [self.feature_names[i] for i in indices]

        return Dataset(
            X=X,
            y=self.y,
            task=self.task,
            name=self.name,
            feature_names=feature_names,
            target_name=self.target_name,
        )

    def add_feature(self, feature_column, name=None):
        feature_column = np.array(feature_column).reshape(-1, 1)

        if len(feature_column) != self.n_samples:
            raise ValueError("Feature column must match number of samples")

        X = np.hstack([self.X, feature_column])

        feature_names = None
        if self.feature_names is not None:
            feature_names = self.feature_names + [name or "new_feature"]

        return Dataset(
            X=X,
            y=self.y,
            task=self.task,
            name=self.name,
            feature_names=feature_names,
            target_name=self.target_name,
        )

    # ------------------------
    # Conversion helpers
    # ------------------------
    def to_numpy(self):
        return self.X, self.y

    def to_list(self):
        return self.X.tolist(), self.y.tolist() if self.y is not None else None

    # ------------------------
    # Representation
    # ------------------------
    def __repr__(self):
        return (
            f"Dataset(name={self.name}, task={self.task}, "
            f"samples={self.n_samples}, features={self.n_features})"
        )