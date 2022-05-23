import numpy as np

"""
CODE TAKEN FROM
https://github.com/alextimans/t4c2021-uncertainty-thesis/
"""

def ence(pred):

   """
   Receives: prediction tensor (samples, 3, 6, H, W, Ch), where 2nd dim
   '3' is y_true (0), point prediction (1), uncertainty measure (2).
   Returns: Expected normalized calibration error (ENCE) across the 
   sample dimension as tensor (6, H, W, Ch).
   Uncertainty measure is assumed to be a standard deviation.
   Every sample is treated as its own individual "bin".
   Then ENCE = mean(|std - rse| / std), mean over samples.
   """

   return np.mean(np.abs(pred[:, 2, ...] - np.sqrt((pred[:, 0, ...] - pred[:, 1, ...])**2)) / pred[:, 2, ...], axis=0)



def get_quantile(pred, n: int = None, alpha: float = 0.1):

    """
    Receives: prediction tensor (samples, 3, 6, H, W, Ch), where 2nd dim
    '3' is y_true (0), point prediction (1), uncertainty measure (2).
    Returns: the conformal prediction score function quantile across
    the sample dimension as tensor (6, H, W, Ch).
    Prediction tensor should contain all predictions for the calibration set
    on a single city (may be large in the first dim depending on calibration set size).
    n: int
        Size of the calibration set.
    alpha: float
        Desired coverage of prediction interval is 1 - alpha, thus governing quantile selection.
    """

    n = n if n is not None else pred.shape[0]
    quant = np.ceil((1 - alpha) * (n + 1)) / n

    return np.quantile(np.abs(pred[:, 0, ...] - pred[:, 1, ...]) / pred[:, 2, ...], quant, axis=0)

def get_pred_interval(pred, quantiles):

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is point prediction (0), uncertainty measure (1);
    quantile tensor (6, H, W, Ch) with calibration set quantiles.
    Returns: prediction interval tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is interval lower bound (0), interval upper bound (1).
    The prediction intervals returned are symmetric about the prediction.
    
    Prediction tensor should contain the predictions for the test set on 
    a single city that matches the city for which the quantiles were computed.
    Note: Interval values are not clamped to [0, 255] and thus may exceed uint8 limits.
          Clamping will influence mean PI width metric if performed prior to evaluation.
    """

    return np.stack(((pred[:, 0, ...] - pred[:, 1, ...] * quantiles),
                        (pred[:, 0, ...] + pred[:, 1, ...] * quantiles)), axis=1)


def coverage(pred_intervals, y_true):

    """
    Receives: prediction interval tensor (samples, 3, 6, H, W, Ch), where 2nd dim
    '3' is y_true(0), interval lower bound (1), interval upper bound (2).
    Returns: empirical coverage as a fraction across the sample dimension
    as tensor (6, H, W, Ch) with values in [0, 1].
    """

    bool_mask = np.stack(((y_true >= pred_intervals[:, 0, ...]),
                             (y_true <= pred_intervals[:, 1, ...])), axis= 1)

    bool_mask = np.ones_like(bool_mask[:, 0, ...]) * (np.sum(bool_mask, axis=1) > 1)

    return (np.sum(bool_mask, axis=0) / bool_mask.shape[0])


def mean_pi_width(pred):

    """
    Receives: prediction interval tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is interval lower bound (0), interval upper bound (1).
    Returns: prediction interval width mean across the sample dimension as
    tensor (6, H, W, Ch).
    """

    return np.mean(np.abs(pred[:, 1, ...] - pred[:, 0, ...]), axis=0)