"""
trainer.py
----------
Training loop logic shared across RankNet and LambdaRank.

Functions:
    train_epoch(model, optimizer, data, config)   : One training epoch
    evaluate(model, data, k)                      : Evaluate NDCG@k on a split
    run_training(model, train_data, val_data, config) : Full training loop
"""


def train_epoch(model, optimizer, data, config):
    raise NotImplementedError("To be implemented")


def evaluate(model, data, k=10):
    raise NotImplementedError("To be implemented")


def run_training(model, train_data, val_data, config):
    raise NotImplementedError("To be implemented")
