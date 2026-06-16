"""
helpers.py
----------
Reproducibility and general utility functions.

Functions:
    set_seed(seed)          : Set random seeds for numpy, torch, python random
    get_device()            : Return CUDA device if available, else CPU
    save_results(results, path) : Save experiment results dict to JSON
"""

import random
import numpy as np


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return best available device."""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"


def save_results(results, path):
    """Save results dictionary as JSON."""
    import json
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
