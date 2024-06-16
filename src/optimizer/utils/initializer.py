# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.

"""
Initializers code.
"""

import typing as tp

import numpy as np
import numpy.typing as npt
from sklearn.utils import check_random_state

from optimizer.types import Vector


def latin_hypercube(
    shape: tp.Tuple[int, int], rng: tp.Union[int, None, np.random.RandomState] = None
) -> Vector:
    """Uses latin hypercube sampling from [0, 1)^d.
    See here for more: \
    https://en.wikipedia.org/wiki/Latin_hypercube_sampling

    Code taken from: \
    scipy.optimize._differentialevolution.DifferentialEvolutionSolver.

    Args:
        shape: shape of the problem, (n_samples, n_dimensions).
        rng: supports the same typing as sklearn.utils.check_random_state.

            - int: seed for a new RandomState.
            - np.RandomState, will be used as the random generator.
            - None: the current RandomState will be used.

    Returns:
        np.ndarray with the specified shape.
    """
    random_state: np.random.RandomState = check_random_state(rng)

    n_samples, n_dimensions = shape

    # Split the space of [0, 1) into equal length segments.
    segment_size = 1.0 / n_samples

    # Sample uniformly sample from [0, segment_size)^(n x d).
    samples = segment_size * random_state.random_sample((n_samples, n_dimensions))

    # Offset to be in the range [0, 1)^(n x d).
    # First row will be in [0, segment_size)^d.
    # Second will be [segment_size, 2 * segment_size)^d, and so on.
    samples += np.linspace(0.0, 1.0, n_samples, endpoint=False)[:, np.newaxis]

    output: Vector = np.zeros_like(samples)

    # Create output by randomly permuting each column of the offset samples.
    for column in range(n_dimensions):
        order = random_state.permutation(n_samples)
        output[:, column] = samples[order, column]

    return output


def uniform_random(
    shape: tp.Tuple[int, ...], rng: tp.Union[int, None, np.random.RandomState] = None
) -> Vector:
    """Uniformly samples from [0, 1)^d.

    Args:
        shape: shape of the problem, (n_samples, n_dimensions).
        rng: supports the same typing as sklearn.utils.check_random_state.
            - int: seed for a new RandomState.
            - np.RandomState, will be used as the random generator.
            - None: the current RandomState will be used.

    Returns:
        np.ndarray with the specified shape.
    """
    random_state: np.random.RandomState = check_random_state(rng)

    return tp.cast(npt.NDArray["np.generic"], random_state.random_sample(shape))
