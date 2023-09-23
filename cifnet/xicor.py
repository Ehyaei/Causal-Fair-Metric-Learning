"""
This code adopter from https://github.com/rtlemos/nonlinear_correlation
"""
import numpy as np
import torch
from scipy.stats import norm


def xicor(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Computes Sourav Chatterjee's nonlinear correlation coefficient for continuous variables

    :param x: sample of predictor variable
    :param y: sample of response variable (same length as x)
    :return: dict with correlation coefficient value and asymptotic p-value, assuming no ties
    """
    _check_inputs(x, y)
    rank_y = _get_rank(y[np.argsort(x)])
    anti_rank_y = _get_anti_rank(rank_y)
    numerator = _get_numerator(rank_y)
    denominator = _get_denominator(anti_rank_y)
    xi = 1 - numerator / denominator
    p_value = _nonlinear_p_value(xi, len(x))
    return {'correlation': xi, 'p_value': p_value}


def _check_inputs(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) != len(y):
        raise ValueError('the two arrays have different lengths: ' + str(len(x)) + ' vs ' + str(len(y)))


def _get_rank(z: np.ndarray) -> np.ndarray:
    temp = np.argsort(z)
    ranks = np.empty_like(temp)
    ranks[temp] = 1 + np.arange(len(z))
    return ranks


def _get_anti_rank(rank_y: np.ndarray) -> np.ndarray:
    return len(rank_y) - rank_y + 1


def _get_numerator(rank_y: np.ndarray) -> np.ndarray:
    return len(rank_y) * np.sum([np.abs(r_next - r) for r_next, r in zip(rank_y[1:], rank_y[:-1])])


def _get_denominator(antirank_y: np.ndarray) -> np.ndarray:
    return 2 * np.sum(antirank_y * (len(antirank_y) - antirank_y))


def _nonlinear_p_value(xi: float, n: int) -> float:
    return 1 - norm.cdf(xi * np.sqrt(n * 5 / 2))


def xicorrcoef(x):
    n, k = x.shape

    m = np.zeros((k, k), np.float32)

    for i in range(k):
        for j in range(i, k):
            m[i, j] = xicor(x[:, i], x[:, j])['correlation']
            m[j, i] = m[i, j]

    return m