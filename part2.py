
"""Part 2: Bagging and multiclass AdaBoost.SAMME."""

import numpy as np

from common import (
    DecisionTree,
    MajorityVoteEnsemble,
    load_dataset,
    stratified_split,
    accuracy,
    print_acc,
    species_family_mapping,
    derived_family_predictions,
)


class BaggingClassifier:
    def __init__(self, n_estimators=25, criterion="gini", rng_seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.rng_seed = rng_seed
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.rng_seed)

        self.trees = []
        n = len(y)

        for _ in range(self.n_estimators):
            # Draw a bootstrap sample of the original training set.
            indices = rng.randint(0, n, size=n)

            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=None,
                min_samples_split=2,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

        self.voter_ = MajorityVoteEnsemble(self.trees, self.classes_)
        return self

    def predict(self, X):
        return self.voter_.predict(X)


class AdaBoostSAMME:
    def __init__(
        self,
        n_estimators=50,
        criterion="gini",
        max_depth=3,
        rng_seed=42,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.rng_seed = rng_seed

        self.trees = []
        self.alphas = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n = len(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        # Start with uniform sample weights.
        weights = np.full(n, 1.0 / n, dtype=float)

        # Keep a reproducible ensemble RNG.
        self.rng_ = np.random.RandomState(self.rng_seed)

        self.trees = []
        self.alphas = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=2,
            )

            # Train the weak learner using the current sample weights.
            tree.fit(X, y, sample_weight=weights)

            pred = tree.predict(X)
            incorrect = pred != y

            err = float(np.sum(weights * incorrect))

            # Prevent numerical issues in the SAMME weight calculation.
            err = float(np.clip(err, 1e-10, 1.0 - 1e-10))

            # Required SAMME stopping rule:
            # discard this learner and stop if its error is too high.
            if err >= 1.0 - 1.0 / K:
                break

            # Multiclass SAMME learner weight.
            alpha = np.log((1.0 - err) / err) + np.log(K - 1)

            # Retain only learners that pass the error threshold.
            self.trees.append(tree)
            self.alphas.append(float(alpha))

            # Increase the weights of misclassified samples.
            weights *= np.exp(alpha * incorrect.astype(float))

            weights_sum = weights.sum()

            if not np.isfinite(weights_sum) or weights_sum <= 0:
                raise FloatingPointError(
                    "AdaBoost sample weights became invalid."
                )

            # Normalize weights for the next boosting round.
            weights /= weights_sum

            # Stop when the learner makes effectively no weighted errors.
            if err <= 1e-10:
                break

        if not self.trees:
            raise RuntimeError(
                "AdaBoost could not retain any weak learner."
            )

        return self

    def predict(self, X):
        X = np.asarray(X)

        scores = np.zeros(
            (len(X), len(self.classes_)),
            dtype=float,
        )

        class_to_index = {
            cls: i for i, cls in enumerate(self.classes_.tolist())
        }

        for tree, alpha in zip(self.trees, self.alphas):
            pred = tree.predict(X)

            for cls in self.classes_:
                scores[:, class_to_index[cls]] += alpha * (pred == cls)

        return self.classes_[np.argmax(scores, axis=1)]


def report(name, model, Xtr, ytr, Xte, yte, fte, mapping):
    pred_train = model.predict(Xtr)
    pred_test = model.predict(Xte)

    print(f"\n{name}")

    print_acc(
        "  Species train accuracy",
        ytr,
        pred_train,
    )

    print_acc(
        "  Species test accuracy",
        yte,
        pred_test,
    )

    print_acc(
        "  Derived family test accuracy",
        fte,
        derived_family_predictions(pred_test, mapping),
    )

    return accuracy(yte, pred_test)


def main():
    X, species, family, _ = load_dataset()

    # Use the assignment-prescribed shared stratified split.
    train_idx, test_idx = stratified_split(species)

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = species[train_idx], species[test_idx]
    fte = family[test_idx]

    mapping = species_family_mapping(
        ytr,
        family[train_idx],
    )

    print("PART 2: BAGGING AND ADABOOST SAMME")

    for criterion in ("gini", "entropy"):
        # Bagging: 25 bootstrap decision trees.
        bag = BaggingClassifier(
            n_estimators=25,
            criterion=criterion,
        ).fit(Xtr, ytr)

        report(
            f"Bagging | {criterion} | 25 trees",
            bag,
            Xtr, ytr,
            Xte, yte,
            fte, mapping,
        )

        # AdaBoost SAMME: 50 rounds, depth-3 weak learners.
        boost3 = AdaBoostSAMME(
            n_estimators=50,
            criterion=criterion,
            max_depth=3,
        ).fit(Xtr, ytr)

        report(
            f"AdaBoost SAMME | {criterion} | depth=3 "
            f"| retained={len(boost3.trees)}",
            boost3,
            Xtr, ytr,
            Xte, yte,
            fte, mapping,
        )

        # Additional depth-1 stump comparison.
        boost1 = AdaBoostSAMME(
            n_estimators=50,
            criterion=criterion,
            max_depth=1,
        ).fit(Xtr, ytr)

        report(
            f"AdaBoost SAMME | {criterion} | stumps depth=1 "
            f"| retained={len(boost1.trees)}",
            boost1,
            Xtr, ytr,
            Xte, yte,
            fte, mapping,
        )


if __name__ == "__main__":
    main()
