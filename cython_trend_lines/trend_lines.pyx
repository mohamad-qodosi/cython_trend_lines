# distutils: language = c++

import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uint32_t
from cython.parallel cimport prange

cimport cython
cimport numpy as np

cdef extern from "covariance.h" namespace "regressor":
    cdef struct covmeans:
        double x
        double y
    
    cdef struct covs:
        covmeans avg
        double s_xx
        double s_xy
        double s_yx
        double s_yy
        uint32_t size
    
    covs covariance(double * x, const uint32_t & size_x, double * y, const uint32_t & size_y) except +


def linregress_simple(double[::1] x, double[::1] y):
    ''' perform simple linear regression on two float_t numpy arrays
    
    Args:
        x: numpy array of x-values (as numpy.float_t dtype)
        y: numpy array of y-values (as numpy.float_t dtype)
    
    Returns:
        LinregressResult with slope, intercept, r-value, p-value and standard error
    '''
    vals = covariance(&x[0], len(x), &y[0], len(y))
    
    # s_xx = vals.s_xx if vals.s_xx != 0 else float('nan')
    # the remainder is from the scipy.stats.linregress function
    cdef double r_num = vals.s_xy
    cdef double s_xx = vals.s_xx
    if s_xx == 0.0:
        s_xx = np.float64('nan')
    cdef double r_den = sqrt(s_xx * vals.s_yy)
    cdef double r = 0.0 if r_den == 0.0 else r_num / r_den
    
    # test for numerical error propagation
    if r > 1.0:
        r = 1.0
    if r < -1.0:
        r = -1.0

    cdef double slope = r_num / s_xx
    cdef double intercept = vals.avg.y - slope * vals.avg.x
    
    return  slope, intercept


cdef np.ndarray[np.float64_t] high_trend_lines_at_last_timestamp(np.ndarray[np.float64_t] high):
    cdef Py_ssize_t default_len = len(high)
    cdef np.ndarray[np.float64_t] x = np.arange(default_len, dtype=np.float64)
    cdef np.ndarray[long] indices
    cdef double slope
    cdef double intercept
    x = np.delete(x, np.isnan(high))
    high = np.delete(high, np.isnan(high))

    if len(x) == 0:
        high_trend = np.array([], dtype=np.float64)
    else:
        while len(high) > 10:
            slope, intercept = linregress_simple(x=x, y=high)
            indices = np.where(high < slope * x + intercept)[0]
            if len(indices) == 0 or (len(indices) >= len(high)):
                break
            x = np.delete(x, indices)
            high = np.delete(high, indices)

        slope, intercept = linregress_simple(x=x, y=high)
        high_trend = np.array([slope, intercept], dtype=np.float64)
    return high_trend

cdef np.ndarray[np.float64_t] low_trend_lines_at_last_timestamp(np.ndarray[np.float64_t] low):
    cdef Py_ssize_t default_len = len(low)
    cdef np.ndarray[np.float64_t] x = np.arange(default_len, dtype=np.float64)
    cdef np.ndarray[long] indices
    cdef double slope
    cdef double intercept
    x = np.delete(x, np.isnan(low))
    low = np.delete(low, np.isnan(low))

    if len(x) == 0:
        low_trend = np.array([], dtype=np.float64)
    else:
        while len(low) > 10:
            slope, intercept = linregress_simple(x=x, y=low)
            indices = np.where(low > slope * x + intercept)[0]
            if len(indices) == 0 or (len(indices) >= len(low)):
                break
            x = np.delete(x, indices)
            low = np.delete(low, indices)

        slope, intercept = linregress_simple(x=x, y=low)
        low_trend = np.array([slope, intercept], dtype=np.float64)
    return low_trend

@cython.boundscheck(False)
@cython.wraparound(False)
def high_trend_lines(high_values: np.ndarray, lookback_freq: int, dont_look_freq: int):
    high = np.pad(high,((lookback_freq - 1, 0)), constant_values=np.nan)
    values = np.empty(len(high_values), dtype=np.float64)
    cdef np.ndarray[np.float64_t] high_trend

    for idx in range(high_values.shape[0] - lookback_freq + 1):
        window = high_values[idx:idx + lookback_freq]
        high_trend = high_trend_lines_at_last_timestamp(window[:lookback_freq - dont_look_freq])
        
        if high_trend.size == 0:
            values[idx] = np.nan
        else:
            values[idx] = high_trend[0] * lookback_freq + high_trend[1] - window[lookback_freq - 1]

    return values

@cython.boundscheck(False)
@cython.wraparound(False)
def low_trend_lines(low_values: np.ndarray, lookback_freq: int, dont_look_freq: int):
    low = np.pad(low,((lookback_freq - 1, 0)), constant_values=np.nan)
    values = np.empty(len(low_values), dtype=np.float64)
    cdef np.ndarray[np.float64_t] low_trend

    for idx in range(low_values.shape[0] - lookback_freq + 1):
        window = low_values[idx:idx + lookback_freq]
        low_trend = low_trend_lines_at_last_timestamp(window[:lookback_freq - dont_look_freq])
        
        if low_trend.size == 0:
            values[idx] = np.nan
        else:
            values[idx] = low_trend[0] * lookback_freq + low_trend[1] - window[lookback_freq - 1]

    return values