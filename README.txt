BITS F464 Machine Learning Assignment 1 - Flat vs. Hierarchical Classification From Scratch

Requirements
- Python 3
- NumPy
- pandas
- Optional: matplotlib (not required by these scripts)
- Do not install/import scikit-learn, SciPy, PyTorch, TensorFlow, XGBoost, LightGBM, statsmodels, or similar ML libraries.

Dataset setup
1. Download the Anuran Calls (MFCCs) dataset from the UCI Machine Learning Repository.
2. Place Frogs_MFCCs.csv in this folder beside the Python scripts.
3. Run commands from this folder. No absolute data paths are used.

Commands (PowerShell or Command Prompt)
python part1_decision_tree.py
python part2_bagging_boosting.py
python part3_random_forest.py
python part4_hierarchical.py
python part5_naive_bayes.py

Implementation notes
- common.py contains the exact assignment stratified split, metrics, confusion matrix, label mapping, and the shared DecisionTree.
- Only MFCCs_1 ... MFCCs_22 are features. Family and Species are labels; Genus and RecordID are never features.
- Test data is only used for final evaluation. Hyperparameters are fixed as specified in the assignment.
- Report the printed train/test species accuracies and test family accuracy in the results matrix. For hierarchical rows, the family accuracy entry is the router accuracy.
- Results are generated at runtime; no accuracy values are hard-coded.

Versions used for your submission
Run these and copy the output into this README before submitting:
python --version
python -c "import numpy, pandas; print('numpy', numpy.__version__); print('pandas', pandas.__version__)"

Important
The assignment says the submitted code must be your own work. Study, test, and adapt this starter implementation so you can explain every design choice and confirm it meets your course's expectations.
