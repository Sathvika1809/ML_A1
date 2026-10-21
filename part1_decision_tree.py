"""Part 1: single decision trees, Gini/entropy, and depth sweep."""
import numpy as np
from common import (DecisionTree, load_dataset, stratified_split, accuracy,
                    print_acc, species_family_mapping, derived_family_predictions)


def main():
    X, species, family, _ = load_dataset()
    train_idx, test_idx = stratified_split(species)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = species[train_idx], species[test_idx]
    ftr, fte = family[train_idx], family[test_idx]
    mapping = species_family_mapping(ytr, ftr)
    print(f"Rows: {len(X)} | train: {len(train_idx)} | test: {len(test_idx)} | features: {X.shape[1]}")
    print("\nPART 1: SINGLE DECISION TREE")
    for criterion in ("gini", "entropy"):
        model = DecisionTree(criterion=criterion, max_depth=None, min_samples_split=2)
        model.fit(Xtr, ytr)
        ptr, pte = model.predict(Xtr), model.predict(Xte)
        print(f"\nCriterion: {criterion}")
        print_acc("  Species train accuracy", ytr, ptr)
        print_acc("  Species test accuracy", yte, pte)
        print_acc("  Derived family test accuracy", fte, derived_family_predictions(pte, mapping))

    print("\nDEPTH SWEEP (for understanding only; not part of results matrix)")
    for criterion in ("gini", "entropy"):
        print(f"\nCriterion: {criterion}")
        print("max_depth | train_species_% | test_species_%")
        for depth in (1, 2, 3, 5, 8, 12, None):
            model = DecisionTree(criterion=criterion, max_depth=depth, min_samples_split=2)
            model.fit(Xtr, ytr)
            tr_acc = 100 * accuracy(ytr, model.predict(Xtr))
            te_acc = 100 * accuracy(yte, model.predict(Xte))
            print(f"{str(depth):>9} | {tr_acc:15.2f} | {te_acc:14.2f}")


if __name__ == "__main__":
    main()
