"""
eval.py
-------
Evaluation metrics for Learning to Rank.

Functions:
    dcg_at_k(labels, k)              : DCG@k for a single ranked list
    ndcg_at_k(labels, k)             : NDCG@k for a single ranked list
    mean_ndcg_at_k(groups, scores, labels, k) : Mean NDCG@k across all queries
    map_score(groups, scores, labels)         : Mean Average Precision
"""

import numpy as np


def dcg_at_k(labels, k):
    """
    Compute Discounted Cumulative Gain at cutoff k.

    Labels must already be in ranked order (highest score first).
    Uses the standard IR formula: (2^rel - 1) / log2(pos + 1)

    Args:
        labels (array-like): Relevance labels in ranked order
        k (int): Cutoff position

    Returns:
        float: DCG@k value
    """
    labels = np.array(labels[:k], dtype=np.float32)
    if len(labels) == 0:
        return 0.0
    gains = (2 ** labels) - 1
    discounts = np.log2(np.arange(2, len(labels) + 2))  # log2(2), log2(3), ...
    return float(np.sum(gains / discounts))


def ndcg_at_k(labels, k):
    """
    Compute Normalized DCG at cutoff k.

    NDCG = DCG@k / IDCG@k
    where IDCG is the DCG of the ideal (perfect) ranking.

    Args:
        labels (array-like): Relevance labels in ranked order
        k (int): Cutoff position

    Returns:
        float: NDCG@k in [0, 1]. Returns 1.0 if no relevant docs exist.
    """
    actual_dcg = dcg_at_k(labels, k)
    # Ideal ranking: sort labels descending
    ideal_labels = sorted(labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_labels, k)
    if ideal_dcg == 0.0:
        return 1.0   # no relevant documents — perfect by convention
    return actual_dcg / ideal_dcg


def mean_ndcg_at_k(groups, scores, labels, k):
    """
    Compute mean NDCG@k across all queries.

    Args:
        groups (list of int): Number of documents per query
        scores (np.ndarray): Model scores, flat array
        labels (np.ndarray): Relevance labels, flat array
        k (int): Cutoff position

    Returns:
        float: Mean NDCG@k across all queries
    """
    ndcg_scores = []
    start = 0
    for count in groups:
        end = start + count
        q_scores = scores[start:end]
        q_labels = labels[start:end]

        # Rank documents by score descending
        ranked_indices = np.argsort(q_scores)[::-1]
        ranked_labels = q_labels[ranked_indices]

        ndcg_scores.append(ndcg_at_k(ranked_labels, k))
        start = end

    return float(np.mean(ndcg_scores))


def map_score(groups, scores, labels):
    """
    Compute Mean Average Precision across all queries.
    Treats labels > 0 as relevant.

    Args:
        groups (list of int): Number of documents per query
        scores (np.ndarray): Model scores, flat array
        labels (np.ndarray): Relevance labels, flat array

    Returns:
        float: MAP score
    """
    ap_scores = []
    start = 0
    for count in groups:
        end = start + count
        q_scores = scores[start:end]
        q_labels = (labels[start:end] > 0).astype(int)  # binary relevance

        ranked_indices = np.argsort(q_scores)[::-1]
        ranked_labels = q_labels[ranked_indices]

        # Average Precision
        hits, precision_sum = 0, 0.0
        for i, rel in enumerate(ranked_labels):
            if rel == 1:
                hits += 1
                precision_sum += hits / (i + 1)
        ap = precision_sum / max(hits, 1)
        ap_scores.append(ap)
        start = end

    return float(np.mean(ap_scores))
