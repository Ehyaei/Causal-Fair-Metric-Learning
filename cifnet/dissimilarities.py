"""
This file contains the dissimilarity functions used in the embedded metric space.
"""
import torch


def l_p(x1, x2, p=2):
    """
    Compute the L_p distance between two tensors of shape [n, d].

    Parameters:
    - tensor1: The first input tensor of shape [n, d].
    - tensor2: The second input tensor of shape [n, d].
    - p: The order of the L_p norm (default is 2 for Euclidean distance).

    Returns:
    - distances: A tensor of shape [n] containing the L_p distances between
      corresponding vectors in tensor1 and tensor2.
    """
    if x1.shape != x2.shape:
        raise ValueError("Input tensors must have the same shape [n, d].")

    # Compute element-wise absolute differences
    absolute_diff = torch.abs(x1 - x2)

    # Compute the L_p norm along axis 1 (dimension d) to get the distances
    if len(absolute_diff.shape) == 1:
        distances = torch.norm(absolute_diff, p=p, dim=0)
    else:
        distances = torch.norm(absolute_diff, p=p, dim=1)

    return distances


l1 = lambda x1, x2: l_p(x1, x2, p=1)
l2 = lambda x1, x2: l_p(x1, x2, p=2)
linf = lambda x1, x2: l_p(x1, x2, p=float('inf'))
l05 = lambda x1, x2: l_p(x1, x2, p=0.5)


def discrete_metric(x, y):
    is_equal = torch.equal(x, y)

    return 1 if not is_equal else 0
