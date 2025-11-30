import scipy
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')


def perc_error_distance(ts, trend):
    """
    Perc. error distance which measures how much the synthetic data follows a trend constraint by evaluating the L2 distance between the TS and the trend.
    """

    if ts.shape != trend.shape:
        raise ValueError("Shape of time series and trend must be the same.")
    
    # Calculate squared differences and sum them
    squared_diff = np.square(ts - trend)
    sum_squared_diff = np.sum(squared_diff, axis=(1, 2))  # sum over seq_length and variables dimensions
    
    # Take the square root of the sum of squared differences
    l2_distances = np.sqrt(sum_squared_diff)
    return np.mean(l2_distances)



def satisfaction_rate(ts, constraint, prior, threshold=1e-2):
    """
    Satisfaction rate which measures the percentage of time a synthetic TS meets the input constraints.
    """
    if ts.shape != constraint.shape:
        raise ValueError("Shape of time series and trend must be the same.")
    

    if 'ohlc' in prior:
        open_t, high_t, low_t, close_t = ts[..., 0], ts[..., 1], ts[..., 2], ts[..., 3]

        # Create masks for conditions where 'open' and 'close' are between 'low' and 'high'
        open_condition = (open_t >= low_t) & (open_t <= high_t)
        close_condition = (close_t >= low_t) & (close_t <= high_t)
        combined_condition = open_condition & close_condition

        satisfaction_rate = np.mean(combined_condition)
        return satisfaction_rate
    
    else:
        mask = constraint > 0
        total = np.sum(mask)
        satifscation = np.sum(mask * (np.abs(constraint - ts) < threshold))
        satisfaction_rate = satifscation / total
        return satisfaction_rate
    
    


