"""Part 4: family router + family-specific species experts."""
import numpy as np
from common import (DecisionTree, load_dataset, stratified_split, accuracy, print_acc,
                    confusion_matrix_numpy, print_confusion_matrix)


class HierarchicalClassifier:
    def __init__(self, router, experts):
        self.router, self.experts = router, experts

    def predict_family(self, X):
        return self.router.predict(X)

    def predict(self, X):
        X = np.asarray(X)
        routed_family = self.predict_family(X)
        result = np.empty(len(X), dtype=next(iter(self.experts.values())).classes_.dtype)
        # Group by predicted family and invoke each relevant expert once.
        for family_name, expert in self.experts.items():
            idx = np.flatnonzero(routed_family == family_name)
            if len(idx):
                result[idx] = expert.predict(X[idx])
        return result

    def predict_oracle(self, X, true_family):
        """Evaluation-only: use the true family to select the expert."""
        X, true_family = np.asarray(X), np.asarray(true_family)
        result = np.empty(len(X), dtype=next(iter(self.experts.values())).classes_.dtype)
        for family_name, expert in self.experts.items():
            idx = np.flatnonzero(true_family == family_name)
            if len(idx):
                result[idx] = expert.predict(X[idx])
        return result


def fit_hierarchical(Xtr, family_train, species_train, criterion):
    router = DecisionTree(criterion=criterion, max_depth=None, min_samples_split=2).fit(Xtr, family_train)
    experts = {}
    for fam in np.unique(family_train):
        idx = np.flatnonzero(family_train == fam)
        experts[fam] = DecisionTree(criterion=criterion, max_depth=None,
                                    min_samples_split=2).fit(Xtr[idx], species_train[idx])
    return HierarchicalClassifier(router, experts)


def report_split(model, X, family, species, split_name, all_families, all_species, print_matrices=False):
    router_pred = model.predict_family(X)
    species_pred = model.predict(X)
    print(f"\n{split_name} metrics")
    print_acc("  Router family accuracy", family, router_pred)
    print_acc("  Overall species accuracy", species, species_pred)
    if print_matrices:
        cm_f, labels_f = confusion_matrix_numpy(family, router_pred, labels=all_families)
        cm_s, labels_s = confusion_matrix_numpy(species, species_pred, labels=all_species)
        print_confusion_matrix(cm_f, labels_f, "Router 4x4 family confusion matrix")
        print_confusion_matrix(cm_s, labels_s, "Final 10x10 species confusion matrix")
        print("\nPer-family final species accuracy:")
        for fam in all_families:
            idx = family == fam
            print(f"  {fam}: {100*accuracy(species[idx], species_pred[idx]):.2f}% (n={np.sum(idx)})")
    return router_pred, species_pred


def main():
    X, species, family, _ = load_dataset()
    train_idx, test_idx = stratified_split(species)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = species[train_idx], species[test_idx]
    ftr, fte = family[train_idx], family[test_idx]
    all_families, all_species = np.unique(family), np.unique(species)
    print("PART 4: HIERARCHICAL DECISION TREE")

    for criterion in ("gini", "entropy"):
        model = fit_hierarchical(Xtr, ftr, ytr, criterion)
        print(f"\nCriterion: {criterion}")
        train_router, train_species_pred = report_split(model, Xtr, ftr, ytr, "TRAIN", all_families, all_species)
        test_router, test_species_pred = report_split(model, Xte, fte, yte, "TEST", all_families, all_species, True)

        # Oracle: test samples are sent to the expert for their TRUE family.
        oracle_species_pred = model.predict_oracle(Xte, fte)
        oracle_acc = accuracy(yte, oracle_species_pred)
        router_correct = test_router == fte
        expert_wrong_given_route = test_species_pred != yte
        pA = float(np.mean(router_correct & expert_wrong_given_route))
        pB = float(np.mean(~router_correct))
        overall = accuracy(yte, test_species_pred)
        router_acc = accuracy(fte, test_router)
        print("\nTEST failure analysis")
        print(f"  Router accuracy A_router: {100*router_acc:.2f}%")
        print(f"  Oracle accuracy A_oracle: {100*oracle_acc:.2f}%")
        print(f"  pA (router right, expert wrong): {100*pA:.2f}%")
        print(f"  pB (router wrong): {100*pB:.2f}%")
        print(f"  A_overall: {100*overall:.2f}%")
        print(f"  A_router - pA: {100*(router_acc-pA):.2f}%")
        print(f"  Numerical check A_overall == A_router - pA: {np.isclose(overall, router_acc-pA, atol=1e-12)}")


if __name__ == "__main__":
    main()
