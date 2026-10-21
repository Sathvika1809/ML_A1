BITS F464 - MACHINE LEARNING
ASSIGNMENT 1: FLAT VS HIERARCHICAL CLASSIFICATION

============================================================
1. ENVIRONMENT REQUIREMENTS
============================================================

Python Version: 3.14.7
NumPy Version: 2.5.3
pandas Version: 3.0.6

Required packages:
- NumPy
- pandas

Install packages if necessary:
python -m pip install numpy pandas

============================================================
2. DATASET INFORMATION
============================================================

Dataset: Anuran Calls MFCCs
Filename: Frogs_MFCCs.csv

Number of samples: 7195
Number of input features: 22 MFCC features
Number of species classes: 10
Number of family classes: 4

Only the 22 MFCC columns are used as input features.
Family, Genus, Species, and RecordID are excluded from
the feature matrix.

Keep Frogs_MFCCs.csv in the same directory as the Python
source files.

============================================================
3. IMPLEMENTED METHODS
============================================================

Part 1: Single Decision Tree
- Gini impurity
- Entropy

Part 2: Bagging and AdaBoost SAMME
- Bagging with 25 bootstrap trees
- AdaBoost SAMME with 50 rounds
- Depth-3 weak learners
- Depth-1 stump comparison

Part 3: Random Forest
- 50 trees
- 4 candidate features per split
- Gini and entropy criteria

Part 4: Hierarchical Decision Tree
- Family-level router
- Family-specific species experts
- Gini and entropy criteria
- Confusion matrices and failure analysis

Part 5: Gaussian Naive Bayes
- Gaussian likelihoods and class priors
- Variance smoothing
- Synthetic sanity check
- Training-feature correlation analysis

The implementation uses Python, NumPy, pandas, and the
Python standard library.

============================================================
4. HOW TO RUN THE PROGRAMS
============================================================

Open PowerShell in the project directory.

Run each program using:

python part1.py
python part2.py
python part3.py
python part4.py
python part5.py

The programs expect the dataset to be available in the
current project directory.

To save output, for example:

python part1.py > part1_output.txt

The shared module common.py contains functionality used
by the individual programs.

============================================================
5. REPORTED RESULTS
============================================================

Part 1 - Decision Tree
Gini:
  Training species accuracy: 100.00%
  Test species accuracy: 94.86%
  Test family accuracy: 96.11%

Entropy:
  Training species accuracy: 100.00%
  Test species accuracy: 95.34%
  Test family accuracy: 96.73%

Part 2 - Bagging
Gini:
  Training species accuracy: 99.97%
  Test species accuracy: 96.73%
  Test family accuracy: 97.57%

Entropy:
  Training species accuracy: 99.98%
  Test species accuracy: 96.87%
  Test family accuracy: 97.50%

Part 2 - AdaBoost SAMME, Depth 3
Gini:
  Training species accuracy: 97.52%
  Test species accuracy: 95.34%
  Test family accuracy: 96.73%

Entropy:
  Training species accuracy: 96.35%
  Test species accuracy: 94.16%
  Test family accuracy: 95.00%

Part 3 - Random Forest
Gini:
  Training species accuracy: 100.00%
  Test species accuracy: 97.57%
  Test family accuracy: 97.98%

Entropy:
  Training species accuracy: 100.00%
  Test species accuracy: 97.36%
  Test family accuracy: 97.98%

Part 4 - Hierarchical Decision Tree
Gini:
  Training species accuracy: 100.00%
  Test species accuracy: 93.82%
  Test router family accuracy: 95.07%

Entropy:
  Training species accuracy: 100.00%
  Test species accuracy: 94.72%
  Test router family accuracy: 95.62%

Part 5 - Gaussian Naive Bayes
  Training species accuracy: 91.47%
  Test species accuracy: 90.34%
  Test family accuracy: 91.94%

The best reported species test accuracy is 97.57%,
achieved by the Gini Random Forest.

These results correspond to the current implementation
and the prescribed train/test split.

============================================================
6. HIERARCHICAL FAILURE ANALYSIS
============================================================

Gini:
  Router accuracy: 95.07%
  Oracle species accuracy: 98.26%
  pA (router correct, expert wrong): 1.25%
  pB (router wrong): 4.93%

Entropy:
  Router accuracy: 95.62%
  Oracle species accuracy: 98.33%
  pA (router correct, expert wrong): 0.90%
  pB (router wrong): 4.38%

Both configurations passed the numerical check:

A_overall = A_router - pA

============================================================
7. ADDITIONAL NOTES
============================================================

The Gaussian Naive Bayes synthetic sanity check passed.

The training-feature correlation matrix has dimensions
22 x 22.

Mean absolute off-diagonal correlation: 0.2962
Maximum absolute off-diagonal correlation: 0.9055

The required results matrix and supplementary hierarchical
failure analysis are provided separately in results_matrix.pdf.

============================================================
8. SUBMISSION FILES
============================================================

The final submission ZIP must use the filename:

<StudentID>_A1.zip

Include the required source files:

part1_decision_tree.py
part2_bagging_boosting.py
part3_random_forest.py
part4_hierarchical.py
part5_naive_bayes.py
common.py
README.txt
results_matrix.pdf

The Python source files must run from the directory
containing the dataset, without absolute file paths.

============================================================
END OF README
============================================================