"""
callbacks.py
------------
Training utilities: early stopping and metric logging.

Classes:
    EarlyStopping : Stops training when validation metric stops improving
    MetricLogger  : Tracks and saves metrics across epochs
"""


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        raise NotImplementedError("To be implemented")


class MetricLogger:
    def __init__(self, save_path=None):
        raise NotImplementedError("To be implemented")
