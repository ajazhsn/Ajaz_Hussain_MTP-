"""
lambdarank.py
-------------
LambdaRank: Extends RankNet by scaling gradients (lambdas) by |delta NDCG|.

Reference:
    Burges et al., "Learning to Rank with Nonsmooth Cost Functions", NeurIPS 2006.

Classes:
    LambdaRank : PyTorch model with lambda-weighted gradient computation
"""

import torch
import torch.nn as nn


class LambdaRank(nn.Module):
    """
    LambdaRank model: same architecture as RankNet, but gradients during
    backprop are scaled by the change in NDCG from swapping document pairs.

    Args:
        input_dim (int): Number of input features (46 for LETOR 4.0)
        hidden_dim (int): Size of hidden layers
        n_layers (int): Number of hidden layers
    """

    def __init__(self, input_dim=46, hidden_dim=128, n_layers=2):
        super(LambdaRank, self).__init__()
        raise NotImplementedError("To be implemented")

    def forward(self, x):
        raise NotImplementedError("To be implemented")

    def compute_lambdas(self, scores, labels, k=10):
        """
        Compute lambda gradients scaled by |delta NDCG@k|.

        Args:
            scores (Tensor): Model scores for all docs in a query
            labels (Tensor): Relevance labels
            k (int): Cutoff for NDCG computation

        Returns:
            lambdas (Tensor): Gradient signal for each document
        """
        raise NotImplementedError("To be implemented")
