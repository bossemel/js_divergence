import numpy as np
import warnings
import argparse


def js_divergence(prob_x_in_p: np.ndarray, prob_x_in_q: np.ndarray,
                  prob_y_in_p: np.ndarray, prob_y_in_q: np.ndarray, eps=1e-07) -> float:
    """Calculate JS-Divergence using Monte Carlo.
    Params:
        prob_x_in_p: p(x), x from distr p(x), array
        prob_x_in_q: q(x), x from distr p(x), array
        prob_y_in_p: p(y), y from distr q(y), array
        prob_y_in_q: p(y), y from distr q(y), array
    Returns:
        divergence: int, JS-Divergence
    """
    assert prob_x_in_p.shape[0] == prob_x_in_q.shape[0]
    assert prob_x_in_q.shape[0] == prob_y_in_p.shape[0]
    assert prob_y_in_p.shape[0] == prob_y_in_q.shape[0]
    mix_X = prob_x_in_p + prob_x_in_q
    mix_Y = prob_y_in_p + prob_y_in_q

    prob_x_in_p[prob_x_in_p == 0] = 0 + eps
    prob_y_in_q[prob_y_in_q == 0] = 0 + eps

    assert np.min(mix_X) > 0
    assert np.min(mix_Y) > 0

    KL_PM = np.log2((2 * prob_x_in_p) / mix_X)
    KL_PM[mix_X == 0] = 0
    KL_PM = KL_PM.mean()

    KL_QM = np.log2((2 * prob_y_in_q) / mix_Y)
    KL_QM[mix_Y == 0] = 0
    KL_QM = KL_QM.mean()

    divergence = (KL_PM + KL_QM) / 2

    if divergence < 0 - 1e-05:
        warnings.warn("JSD estimate below zero. JSD: {}".format(divergence))

    return divergence
