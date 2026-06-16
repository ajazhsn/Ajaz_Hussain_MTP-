"""
loader.py
---------
Handles loading and parsing of LETOR 4.0 dataset files (SVMlight format).

Format of each line:
    <label> qid:<qid> 1:<f1> 2:<f2> ... 46:<f46> #docid = <docid> ...

Functions:
    load_letor(filepath)      : Loads a single fold file into features, labels, qids
    get_query_groups(qids)    : Returns list of (start_idx, count) per query
"""

import numpy as np


def load_letor(filepath):
    """
    Parse a LETOR 4.0 file in SVMlight format.

    Args:
        filepath (str): Path to the .txt data file (e.g., train.txt)

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, 46)
        y (np.ndarray): Relevance labels of shape (n_samples,)  values in {0, 1, 2}
        qids (np.ndarray): Query IDs of shape (n_samples,)
    """
    labels, qids, features = [], [], []

    with open(filepath, "r") as f:
        for line in f:
            # Strip comment after #
            line = line.split("#")[0].strip()
            if not line:
                continue

            parts = line.split()

            # First token is relevance label
            label = int(parts[0])

            # Second token is qid:xxxxx
            qid = int(parts[1].split(":")[1])

            # Remaining tokens are feature_index:value pairs
            feat = np.zeros(46, dtype=np.float32)
            for token in parts[2:]:
                idx, val = token.split(":")
                feat[int(idx) - 1] = float(val)   # convert 1-indexed to 0-indexed

            labels.append(label)
            qids.append(qid)
            features.append(feat)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    qids = np.array(qids, dtype=np.int32)

    return X, y, qids


def get_query_groups(qids):
    """
    Convert flat query ID array into grouped structure for LTR training.

    Each query group tells the model: "these rows belong to the same query."
    This is essential for pairwise and listwise methods.

    Args:
        qids (np.ndarray): Array of query IDs (must be sorted/contiguous per query)

    Returns:
        groups (list of int): Count of documents per query, in order.
                              e.g. [25, 13, 40, ...] means query 1 has 25 docs,
                              query 2 has 13 docs, query 3 has 40 docs, etc.
    """
    groups = []
    current_qid = qids[0]
    count = 0

    for qid in qids:
        if qid == current_qid:
            count += 1
        else:
            groups.append(count)
            current_qid = qid
            count = 1
    groups.append(count)   # append last group

    return groups
