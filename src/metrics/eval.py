"""
eval.py
-------
Evaluation metrics for Learning to Rank.

Functions:
    dcg_at_k(labels, k)         : Compute DCG@k for a ranked list
    ndcg_at_k(labels, k)        : Compute NDCG@k for a ranked list
    mean_ndcg(all_labels, k)    : Average NDCG@k across all queries
    map_score(all_labels)       : Mean Average Precision
"""

import numpy as np


def dcg_at_k(labels, k):
    """
    Compute Discounted Cumulative Gain at cutoff k.

    Args:
        labels (array-like): Relevance labels in ranked order (descending score)
        k (int): Cutoff position

    Returns:
        float: DCG@k value
    """
    raise NotImplementedError("To be implemented")


def ndcg_at_k(labels, k):
    """
    Compute Normalized DCG at cutoff k.

    Args:
        labels (array-like): Relevance labels in ranked order
        k (int): Cutoff position

    Returns:
        float: NDCG@k value in [0, 1]
    """
    raise NotImplementedError("To be implemented")


def mean_ndcg(all_labels, k):
    """
    Compute mean NDCG@k across multiple queries.

    Args:
        all_labels (list of array-like): Relevance labels per query, in ranked order
        k (int): Cutoff position

    Returns:
        float: Mean NDCG@k
    """
    raise NotImplementedError("To be implemented")


def map_score(all_labels):
    """
    Compute Mean Average Precision across queries.

    Args:
        all_labels (list of array-like): Binary relevance labels per query, in ranked order

    Returns:
        float: MAP score
    """
    raise NotImplementedError("To be implemented")
