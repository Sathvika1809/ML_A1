# Flat vs. Hierarchical Classification From Scratch

**Course:** BITS F464 --- Machine Learning\
**Assignment:** Assignment 1\
**Language:** Python 3

## Overview

This project compares flat and hierarchical classification methods
implemented from scratch on the **Anuran Calls MFCCs** dataset. The
dataset contains 7,195 examples, 22 MFCC features, and the labels
`Family`, `Genus`, and `Species`. `RecordID` is an identifier, not a
predictive feature.

The implementation uses Python, NumPy, pandas, and the Python standard
library. Machine-learning helper libraries such as scikit-learn,
PyTorch, TensorFlow, SciPy, and statsmodels are not used.

## Methods

### Part 1 --- Single Decision Tree

-   Trains an unconstrained decision tree using Gini impurity and
    entropy.
-   Reports training and test species accuracy and derived family
    accuracy.
-   Includes an optional depth sweep for exploration. Use the prescribed
    unconstrained-tree result in the main results table if required by
    the assignment.

### Part 2 --- Bagging and AdaBoost SAMME

-   Bagging with 25 bootstrap decision trees.
-   AdaBoost SAMME with 50 rounds and depth-3 weak learners.
-   Includes a depth-1 stump comparison in the reported run.
-   Compares Gini and entropy criteria.

### Part 3 --- Random Forest

-   Trains 50 trees and samples 4 candidate features at each split.
-   Compares Gini and entropy criteria.
-   Reports species and derived-family accuracy. Mean pairwise
    test-prediction agreement is an additional diagnostic.

### Part 4 --- Hierarchical Decision Tree

-   Uses a family-level router followed by a family-specific species
    expert.
-   Compares Gini and entropy criteria.
-   Reports router accuracy, overall species accuracy, confusion
    matrices, per-family accuracy, oracle accuracy, and routing-failure
    analysis.

### Part 5 --- Gaussian Naive Bayes

-   Implements Gaussian Naive Bayes from scratch.
-   Uses variance smoothing based on `1e-9 * max_j Var_train(X_j)`.
-   Reports species and derived-family accuracy.
-   Computes the training-feature correlation matrix as an additional
    analysis.

## Dataset

Place the assignment's Anuran Calls MFCCs CSV in the project directory
and name it:

``` text
Frogs_MFCCs.csv
```

The CSV should contain the 22 MFCC columns and these label/identifier
columns: - `MFCCs_1` through `MFCCs_22` - `Family` - `Genus` -
`Species` - `RecordID`

In the supplied CSV, the first nine MFCC column names may contain a
space (for example, `MFCCs_ 1`). The loader should normalize these names
or handle them explicitly.

Only the 22 MFCC columns should be used as input features. Do not
include `Family`, `Genus`, `Species`, or `RecordID` in the feature
matrix.

## Requirements

-   Python 3
-   NumPy
-   pandas

Install required packages if needed:

``` powershell
python -m pip install numpy pandas
```

Matplotlib is only needed if your implementation uses it for optional
plots.

## Running the scripts

Run each part from the directory containing the scripts and
`Frogs_MFCCs.csv`:

``` powershell
python part1.py
python part2.py
python part3.py
python part4.py
python part5.py
```

To save console output:

``` powershell
python part1.py > part1_output.txt
```

The script names above match the local files used in this project. If
your files have different names, substitute the appropriate names.

## Reported results

These are the results reported from one run using the project's
train/test split. They are experimental results, not guarantees for
other implementations or splits.

### Best reported species test accuracy by part

  ------------------------------------------------------------------------
  Part                  Method /                     Species test accuracy
                        configuration         
  --------------------- --------------------- ----------------------------
  1                     Decision Tree ---                           95.34%
                        Entropy,              
                        unconstrained         

  2                     Bagging --- Entropy,                        96.87%
                        25 trees              

  2                     AdaBoost SAMME ---                          95.34%
                        Gini, depth 3, 50     
                        retained rounds       

  3                     Random Forest ---                       **97.57%**
                        Gini, 50 trees, 4     
                        candidate features    
                        per split             

  4                     Hierarchical Decision                       94.72%
                        Tree --- Entropy      

  5                     Gaussian Naive Bayes                        90.34%
  ------------------------------------------------------------------------

### Additional observations

-   The best reported species test accuracy was **97.57%**, achieved by
    the Gini Random Forest.
-   Bagging with entropy achieved **96.87%** species test accuracy.
-   The hierarchical entropy model achieved **94.72%** overall species
    accuracy and **95.62%** router family accuracy.
-   Gaussian Naive Bayes achieved **90.34%** species test accuracy.
-   Gaussian Naive Bayes correlation analysis reported a mean absolute
    off-diagonal correlation of `0.2962` and a maximum absolute
    correlation of `0.9055`. This suggests some MFCC features are
    strongly linearly related, but these overall correlations do not by
    themselves establish conditional dependence within each class.

Accuracy can vary with the dataset, split, random seeds, tie-breaking,
and implementation. Keep the assignment-prescribed split and
hyperparameters for the final comparison, and do not use test accuracy
to tune model settings.

## Implementation and evaluation notes

-   Use only MFCC columns as input features.
-   Keep the prescribed stratified train/test split consistent across
    methods.
-   Fit models on training data and use the test set for final
    evaluation.
-   Report the required configurations rather than selecting a model
    solely because it has the highest test accuracy.
-   Confusion matrices should use a consistent label order, with rows
    representing true labels and columns representing predicted labels.
-   A small synthetic Gaussian Naive Bayes sanity check is useful, but
    it does not replace validation against the assignment specification.

## Project structure

``` text
MLAssignment/
├── common.py
├── part1.py
├── part2.py
├── part3.py
├── part4.py
├── part5.py
├── Frogs_MFCCs.csv
└── README.md
```

If your folder contains additional output files or helper modules,
retain them as needed by your implementation.

## Reproducibility checklist

-   [ ] Confirm that the CSV loads and all 22 MFCC features are
    selected.
-   [ ] Confirm that `RecordID` and target labels are excluded from the
    feature matrix.
-   [ ] Confirm that every script uses the prescribed train/test split
    and seed.
-   [ ] Verify tree counts, depth settings, feature subsampling, and
    AdaBoost SAMME update rules.
-   [ ] Verify Gaussian Naive Bayes variance smoothing.
-   [ ] Save final console outputs and include required tables and
    confusion matrices in the report.
-   [ ] Ensure test-set performance was not used to tune the reported
    models.
