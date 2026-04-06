"""
Device detection utilities.
"""

import torch


def get_device() -> str:
    """
    Detect and return the best available device for PyTorch.
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
