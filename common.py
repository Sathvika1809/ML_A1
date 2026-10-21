
from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = "Frogs_MFCCs.csv"
FEATURE_COLUMNS = [f"MFCCs_{i:2d}" if i >= 10 else f"MFCCs_ {i}" 
                   for i in range(1, 23)]


def stratified_split(labels, test_frac=0.2, seed=42):
    """Exact split function prescribed by the assignment."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[:int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test, dtype=int))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test


def load_dataset(path=DATA_FILE):
    """Load only the 22 MFCC columns as X; all labels remain separate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put Frogs_MFCCs.csv in the same folder as this script."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + ["Family", "Species"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    species = df["Species"].to_numpy()
    family = df["Family"].to_numpy()
    return X, species, family, df


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def print_acc(label, y_true, y_pred):
    print(f"{label}: {100 * accuracy(y_true, y_pred):.2f}%")


def confusion_matrix_numpy(y_true, y_pred, labels=None):
    """Confusion matrix with true labels as rows and predicted labels as columns."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[index[yt], index[yp]] += 1
    return cm, labels


def print_confusion_matrix(cm, labels, title):
    """Print a confusion matrix with clearly aligned numeric columns."""
    cm = np.asarray(cm)

    print(f"\n{title} (rows=true, columns=predicted):")
    print("Labels:", " | ".join(map(str, labels)))

    column_width = max(
        5,
        max(len(str(int(value))) for value in cm.flat) + 2,
    )

    for row in cm:
        print(" ".join(f"{int(value):>{column_width}d}" for value in row))




def species_family_mapping(train_species, train_family):
    """Build mapping from training labels only and check one-family-per-species."""
    mapping = {}
    for sp, fam in zip(train_species, train_family):
        if sp in mapping and mapping[sp] != fam:
            raise ValueError(f"Species {sp!r} maps to more than one family in training data")
        mapping[sp] = fam
    return mapping


def derived_family_predictions(species_predictions, mapping):
    missing = sorted(set(np.asarray(species_predictions).tolist()) - set(mapping))
    if missing:
        raise ValueError(f"No training-derived family mapping for species: {missing}")
    return np.array([mapping[s] for s in species_predictions])


class TreeNode:
    __slots__ = ("prediction", "feature", "threshold", "left", "right")

    def __init__(self, prediction, feature=None, threshold=None, left=None, right=None):
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTree:
    """Binary CART-style decision tree with vectorised threshold scoring."""
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, rng=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.rng = rng
        self.root = None
        self.classes_ = None
        self.n_features_in_ = None

    def _impurity(self, class_weights):
        total = class_weights.sum()
        if total <= 0:
            return 0.0
        p = class_weights / total
        if self.criterion == "gini":
            return float(1.0 - np.dot(p, p))
        positive = p > 0
        return float(-np.sum(p[positive] * np.log2(p[positive])))

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2D, non-empty, and have the same row count as y")
        self.n_features_in_ = X.shape[1]
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if sample_weight is None:
            weights = np.full(len(y), 1.0 / len(y), dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float).copy()
            if len(weights) != len(y) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("sample_weight must be nonnegative, aligned with y, and have positive sum")
            weights /= weights.sum()
        self._n_classes = len(self.classes_)
        self._X = X
        self._y = y_encoded
        self._weights = weights
        self.root = self._build(np.arange(len(y)), depth=0)
        del self._X, self._y, self._weights
        return self

    def _build(self, indices, depth):
        y = self._y[indices]
        w = self._weights[indices]
        class_weights = np.bincount(y, weights=w, minlength=self._n_classes)
        # np.argmax returns the first maximum, hence smallest sorted class wins ties.
        prediction = self.classes_[int(np.argmax(class_weights))]
        node = TreeNode(prediction)
        if (np.count_nonzero(class_weights > 0) <= 1 or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(indices) < self.min_samples_split):
            return node

        parent_impurity = self._impurity(class_weights)
        if parent_impurity <= 0:
            return node

        d = self._X.shape[1]
        if self.max_features is None:
            features = np.arange(d)
        else:
            m = min(int(self.max_features), d)
            if self.rng is None:
                raise ValueError("Pass rng when max_features is set")
            features = np.sort(self.rng.choice(d, m, replace=False))

        total_weight = w.sum()
        best_gain, best_feature, best_threshold = 0.0, None, None
        best_left = best_right = None
        for feature in features:
            vals = self._X[indices, feature]
            order = np.argsort(vals, kind="mergesort")
            sv, sy, sw = vals[order], y[order], w[order]
            # Candidate cuts only occur between distinct adjacent values.
            valid = sv[:-1] < sv[1:]
            if not np.any(valid):
                continue
            # Cumulative class weights: rows are sorted observations, columns are classes.
            one_hot_weight = np.zeros((len(indices), self._n_classes), dtype=float)
            one_hot_weight[np.arange(len(indices)), sy] = sw
            cum = np.cumsum(one_hot_weight, axis=0)
            left_counts = cum[:-1][valid]
            right_counts = class_weights[None, :] - left_counts
            left_w = left_counts.sum(axis=1)
            right_w = right_counts.sum(axis=1)
            if self.criterion == "gini":
                lp = left_counts / left_w[:, None]
                rp = right_counts / right_w[:, None]
                li = 1.0 - np.sum(lp * lp, axis=1)
                ri = 1.0 - np.sum(rp * rp, axis=1)
            else:
                lp = left_counts / left_w[:, None]
                rp = right_counts / right_w[:, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    lterms = np.where(lp > 0, lp * np.log2(np.where(lp > 0, lp, 1)), 0.0)
                    rterms = np.where(rp > 0, rp * np.log2(np.where(rp > 0, rp, 1)), 0.0)
                li = -np.sum(lterms, axis=1)
                ri = -np.sum(rterms, axis=1)
            gains = parent_impurity - (left_w / total_weight) * li - (right_w / total_weight) * ri
            candidate_positions = np.flatnonzero(valid)
            thresholds = sv[candidate_positions] + (sv[candidate_positions + 1] - sv[candidate_positions]) / 2.0
            # Since features and thresholds are considered in ascending order, updating only
            # for a strictly larger gain implements the required deterministic tie-breaking.
            local_best = int(np.argmax(gains))
            gain = float(gains[local_best])
            if gain > best_gain:
                pos = int(candidate_positions[local_best])
                best_gain, best_feature, best_threshold = gain, int(feature), float(thresholds[local_best])

        if best_feature is None or best_gain <= 0:
            return node
        vals = self._X[indices, best_feature]
        left_mask = vals <= best_threshold
        left_indices, right_indices = indices[left_mask], indices[~left_mask]
        if len(left_indices) == 0 or len(right_indices) == 0:
            return node
        node.feature, node.threshold = best_feature, best_threshold
        node.left = self._build(left_indices, depth + 1)
        node.right = self._build(right_indices, depth + 1)
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.root is None:
            raise RuntimeError("Call fit before predict")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.prediction
        return out


class MajorityVoteEnsemble:
    """Shared majority-vote helper for bagging and random forest."""
    def __init__(self, trees, classes):
        self.trees = trees
        self.classes = np.asarray(classes)

    def predict(self, X):
        if len(self.trees) == 0:
            raise RuntimeError("Ensemble has no fitted trees")
        predictions = np.vstack([tree.predict(X) for tree in self.trees])
        result = []
        for col in predictions.T:
            counts = np.array([np.count_nonzero(col == c) for c in self.classes])
            result.append(self.classes[int(np.argmax(counts))])
        return np.asarray(result, dtype=self.classes.dtype)


def evaluate_flat_model(name, model, X_train, y_train, X_test, y_test, family_test, mapping):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_family = derived_family_predictions(pred_test, mapping)
    print(f"\n{name}")
    print_acc("  Species train accuracy", y_train, pred_train)
    print_acc("  Species test accuracy", y_test, pred_test)
    print_acc("  Derived family test accuracy", family_test, pred_family)
    return pred_train, pred_test, pred_family
