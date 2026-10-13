"""Part 2: bagging and multiclass AdaBoost.SAMME."""
import numpy as np
from common import (DecisionTree, MajorityVoteEnsemble, load_dataset, stratified_split,
                    accuracy, print_acc, species_family_mapping, derived_family_predictions)


class BaggingClassifier:
    def __init__(self, n_estimators=25, criterion="gini", rng_seed=42):
        self.n_estimators, self.criterion, self.rng_seed = n_estimators, criterion, rng_seed
        self.trees, self.classes_ = [], None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.rng_seed)  # one fresh RNG per ensemble
        self.trees = []
        n = len(y)
        for _ in range(self.n_estimators):
            indices = rng.randint(0, n, size=n)
            tree = DecisionTree(criterion=self.criterion, max_depth=None, min_samples_split=2)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        self.voter_ = MajorityVoteEnsemble(self.trees, self.classes_)
        return self

    def predict(self, X):
        return self.voter_.predict(X)


class AdaBoostSAMME:
    def __init__(self, n_estimators=50, criterion="gini", max_depth=3, rng_seed=42):
        self.n_estimators, self.criterion = n_estimators, criterion
        self.max_depth, self.rng_seed = max_depth, rng_seed
        self.trees, self.alphas, self.classes_ = [], [], None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n = len(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        weights = np.full(n, 1.0 / n, dtype=float)
        # Trees themselves are deterministic here; still create the per-ensemble RNG as required.
        self.rng_ = np.random.RandomState(self.rng_seed)
        self.trees, self.alphas = [], []
        for round_no in range(self.n_estimators):
            tree = DecisionTree(criterion=self.criterion, max_depth=self.max_depth,
                                min_samples_split=2)
            tree.fit(X, y, sample_weight=weights)
            pred = tree.predict(X)
            incorrect = pred != y
            err = float(np.sum(weights * incorrect))
            err = float(np.clip(err, 1e-10, 1.0 - 1e-10))
            if err >= 1.0 - 1.0 / K:
                if round_no == 0:
                    # No useful weak learner: preserve a valid fallback classifier.
                    self.trees, self.alphas = [tree], [1.0]
                break
            alpha = np.log((1.0 - err) / err) + np.log(K - 1)
            self.trees.append(tree)
            self.alphas.append(float(alpha))
            weights *= np.exp(alpha * incorrect.astype(float))
            weights_sum = weights.sum()
            if not np.isfinite(weights_sum) or weights_sum <= 0:
                raise FloatingPointError("AdaBoost sample weights became invalid")
            weights /= weights_sum
            if err <= 1e-10:
                break
        if not self.trees:
            raise RuntimeError("AdaBoost could not retain any weak learner")
        return self

    def predict(self, X):
        X = np.asarray(X)
        scores = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_to_index = {c: i for i, c in enumerate(self.classes_.tolist())}
        for tree, alpha in zip(self.trees, self.alphas):
            pred = tree.predict(X)
            for c in self.classes_:
                scores[:, class_to_index[c]] += alpha * (pred == c)
        return self.classes_[np.argmax(scores, axis=1)]


def report(name, model, Xtr, ytr, Xte, yte, fte, mapping):
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    print(f"\n{name}")
    print_acc("  Species train accuracy", ytr, ptr)
    print_acc("  Species test accuracy", yte, pte)
    print_acc("  Derived family test accuracy", fte, derived_family_predictions(pte, mapping))
    return accuracy(yte, pte)


def main():
    X, species, family, _ = load_dataset()
    train_idx, test_idx = stratified_split(species)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = species[train_idx], species[test_idx]
    ftr, fte = family[train_idx], family[test_idx]
    mapping = species_family_mapping(ytr, ftr)
    print("PART 2: BAGGING AND ADABOOST SAMME")
    for criterion in ("gini", "entropy"):
        bag = BaggingClassifier(25, criterion).fit(Xtr, ytr)
        report(f"Bagging | {criterion} | 25 trees", bag, Xtr, ytr, Xte, yte, fte, mapping)
        boost3 = AdaBoostSAMME(50, criterion, max_depth=3).fit(Xtr, ytr)
        report(f"AdaBoost SAMME | {criterion} | depth=3 | retained={len(boost3.trees)}",
               boost3, Xtr, ytr, Xte, yte, fte, mapping)
        boost1 = AdaBoostSAMME(50, criterion, max_depth=1).fit(Xtr, ytr)
        report(f"AdaBoost SAMME | {criterion} | stumps depth=1 | retained={len(boost1.trees)}",
               boost1, Xtr, ytr, Xte, yte, fte, mapping)


if __name__ == "__main__":
    main()
