"""
loader.py
---------
Handles loading and parsing of LETOR 4.0 dataset files (SVMlight format).

Functions:
    load_letor(filepath)      : Loads a single fold file into features, labels, qids
    get_query_groups(qids)    : Returns list of (start_idx, count) per query
"""


def load_letor(filepath):
    """
    Parse a LETOR 4.0 file in SVMlight format.

    Args:
        filepath (str): Path to the .txt data file (e.g., train.txt)

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, 46)
        y (np.ndarray): Relevance labels of shape (n_samples,)
        qids (np.ndarray): Query IDs of shape (n_samples,)
    """
    raise NotImplementedError("To be implemented")


def get_query_groups(qids):
    """
    Convert flat query ID array into grouped structure for LTR training.

    Args:
        qids (np.ndarray): Array of query IDs

    Returns:
        List of (start_index, count) tuples, one per unique query
    """
    raise NotImplementedError("To be implemented")
