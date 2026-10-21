"""Part 5: Gaussian Naive Bayes implemented with NumPy only."""
import numpy as np
from common import (load_dataset, stratified_split, accuracy, print_acc,
                    species_family_mapping, derived_family_predictions)


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        n, d = X.shape
        self.class_prior_ = np.zeros(len(self.classes_), dtype=float)
        self.theta_ = np.zeros((len(self.classes_), d), dtype=float)
        self.var_ = np.zeros((len(self.classes_), d), dtype=float)
        epsilon = self.var_smoothing * np.var(X, axis=0).max()
        for i, cls in enumerate(self.classes_):
            Xc = X[encoded == i]
            self.class_prior_[i] = len(Xc) / n
            self.theta_[i] = np.mean(Xc, axis=0)
            # Population variance (ddof=0), as specified in the assignment.
            self.var_[i] = np.var(Xc, axis=0) + epsilon
        self.epsilon_ = epsilon
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # Broadcasting shapes: X[:, None, :] -> (n,1,d); class parameters -> (1,K,d).
        diff = X[:, None, :] - self.theta_[None, :, :]
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * self.var_[None, :, :]) +
            (diff * diff) / self.var_[None, :, :], axis=2
        )
        log_joint = np.log(self.class_prior_[None, :]) + log_likelihood
        return self.classes_[np.argmax(log_joint, axis=1)]


def sanity_check():
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNB().fit(X, y)
    pred = model.predict(np.array([[3.0], [9.0]]))
    assert np.array_equal(pred, np.array([0, 1])), f"GaussianNB sanity check failed: {pred}"
    print("GaussianNB sanity check passed: x=3 -> class 0; x=9 -> class 1")


def main():
    sanity_check()
    X, species, family, _ = load_dataset()
    train_idx, test_idx = stratified_split(species)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = species[train_idx], species[test_idx]
    ftr, fte = family[train_idx], family[test_idx]
    mapping = species_family_mapping(ytr, ftr)
    model = GaussianNB(var_smoothing=1e-9).fit(Xtr, ytr)
    pred_train, pred_test = model.predict(Xtr), model.predict(Xte)
    print("\nPART 5: GAUSSIAN NAIVE BAYES")
    print(f"  Variance smoothing epsilon: {model.epsilon_:.12g}")
    print_acc("  Species train accuracy", ytr, pred_train)
    print_acc("  Species test accuracy", yte, pred_test)
    print_acc("  Derived family test accuracy", fte, derived_family_predictions(pred_test, mapping))

    corr = np.corrcoef(Xtr, rowvar=False)
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    print("\nTraining-feature correlation analysis")
    print(f"  Correlation matrix shape: {corr.shape}")
    print(f"  Mean absolute off-diagonal correlation: {np.mean(np.abs(off_diagonal)):.4f}")
    print(f"  Maximum absolute off-diagonal correlation: {np.max(np.abs(off_diagonal)):.4f}")
    print("  Full 22x22 training correlation matrix:")
    print(np.array2string(corr, precision=3, suppress_small=True, max_line_width=160))


if __name__ == "__main__":
    main()
