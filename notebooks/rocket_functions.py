# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

import numpy as np
from numba import njit, prange

@njit
def generate_kernels(
    input_length: int, 
    num_kernels: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates random convolutional kernels for the ROCKET algorithm.
    """
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0

    for i in range(num_kernels):
        _length = lengths[i]
        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel(
    X: np.ndarray, 
    weights: np.ndarray, 
    length: int, 
    bias: float, 
    dilation: int, 
    padding: int
    ) -> tuple[float, float]:
    """
    Applies a single random kernel to a time series instance.
    Returns the Proportion of Positive Values (PPV) and the Maximum (Max).
    """
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.inf

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias
        index = i

        for j in range(length):
            if 0 <= index < input_length:
                _sum += weights[j] * X[index]
            index += dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(parallel=True, fastmath=True)
def apply_kernels(
    X: np.ndarray, 
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
    """
    Applies all generated kernels to a dataset of time series in parallel.
    """
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    # 2 features per kernel (PPV and Max)
    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float64) 

    for i in prange(num_examples):
        a1 = 0  # for weights
        a2 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = apply_kernel(
                X[i], 
                weights[a1:b1], 
                lengths[j], 
                biases[j], 
                dilations[j], 
                paddings[j]
            )

            a1 = b1
            a2 = b2

    return _X