"""
splits.py
---------
Utilities for loading the 5 pre-defined LETOR 4.0 cross-validation folds.

Functions:
    load_fold(fold_dir)       : Loads train/vali/test splits for a given fold
    load_all_folds(data_dir)  : Iterates over all 5 folds
"""


def load_fold(fold_dir):
    """
    Load train, validation, and test sets for a single fold.

    Args:
        fold_dir (str): Path to a Fold directory (e.g., .../MQ2007/Fold1/)

    Returns:
        dict with keys 'train', 'vali', 'test', each containing (X, y, qids)
    """
    raise NotImplementedError("To be implemented")


def load_all_folds(data_dir, n_folds=5):
    """
    Generator that yields fold data for all 5 folds.

    Args:
        data_dir (str): Root directory of MQ2007 or MQ2008
        n_folds (int): Number of folds (default: 5)

    Yields:
        dict with keys 'train', 'vali', 'test' for each fold
    """
    raise NotImplementedError("To be implemented")
