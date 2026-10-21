"""Part 3: random forest, sampling four candidate features at every split."""
import numpy as np
from common import (DecisionTree, MajorityVoteEnsemble, load_dataset, stratified_split,
                    print_acc, species_family_mapping, derived_family_predictions)


class RandomForestClassifier:
    def __init__(self, n_estimators=50, criterion="gini", max_features=4, rng_seed=42):
        self.n_estimators, self.criterion = n_estimators, criterion
        self.max_features, self.rng_seed = max_features, rng_seed
        self.trees, self.classes_ = [], None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.rng_seed)  # shared RNG for this forest's bootstrap and split subsets
        self.trees = []
        n = len(y)
        for _ in range(self.n_estimators):
            indices = rng.randint(0, n, size=n)
            tree = DecisionTree(criterion=self.criterion, max_depth=None, min_samples_split=2,
                                max_features=self.max_features, rng=rng)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        self.voter_ = MajorityVoteEnsemble(self.trees, self.classes_)
        return self

    def predict(self, X):
        return self.voter_.predict(X)


def mean_pairwise_agreement(trees, X):
    """Optional: mean fraction of test rows where two trees agree."""
    pred = np.vstack([tree.predict(X) for tree in trees])
    values = []
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            values.append(np.mean(pred[i] == pred[j]))
    return float(np.mean(values)) if values else float("nan")


def main():
    X, species, family, _ = load_dataset()
    train_idx, test_idx = stratified_split(species)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = species[train_idx], species[test_idx]
    ftr, fte = family[train_idx], family[test_idx]
    mapping = species_family_mapping(ytr, ftr)
    print("PART 3: RANDOM FOREST")
    for criterion in ("gini", "entropy"):
        forest = RandomForestClassifier(50, criterion, max_features=4).fit(Xtr, ytr)
        print(f"\nRandom Forest | {criterion} | 50 trees | max_features=4")
        print_acc("  Species train accuracy", ytr, forest.predict(Xtr))
        pred_test = forest.predict(Xte)
        print_acc("  Species test accuracy", yte, pred_test)
        print_acc("  Derived family test accuracy", fte, derived_family_predictions(pred_test, mapping))
        print(f"  Mean pairwise test-prediction agreement: {100*mean_pairwise_agreement(forest.trees, Xte):.2f}% (extra)")


if __name__ == "__main__":
    main()
