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

import typing as tp

import numpy as np

from optimizer.types import MAXIMIZE, MINIMIZE, TBounds, TSense, Vector


def get_best(
    objective_values: Vector, current_best: float, minimize: bool,
) -> TBounds:
    """
    Get the best from a list of objectives and the current best value.

    Args:
        objective_values: Vector of objectives.
        current_best: current minimum objective value.
        minimize: the objective is minimized if true, maximized otherwise

    Returns:
        best found value and its improvement from the current value.
    """
    if minimize:
        return _get_best_minimize(objective_values, current_best)
    else:
        overall_best, improvement = _get_best_minimize(
            np.negative(objective_values), -current_best,
        )
        return -overall_best, improvement


def _get_best_minimize(
    objective_values: Vector, current_best: float
) -> TBounds:
    """ Returns best found value and its improvement from the current value."""
    objectives_best = np.min(objective_values)
    overall_best = np.min([objectives_best, current_best])
    improvement = current_best - overall_best
    return overall_best, improvement


def top_n_indices(x: Vector, sense: TSense, top_n: int) -> Vector:
    """
    Utility for find the indices of the `top_n`
    elements of a vector, when that vector is
    sorted according to `sense`.

    Argsort sorts in ascending order. For maximize,
    we take the last `top_n` elements.
    For minimization, we sort then take the first
    `top_n` elements.

    Args:
        x: Vector of values to sort and find best idx for.
        sense: Whether to maximize or minimize the function
        top_n: How many top solutions to check for constraint violations

    Returns:
        int

    """
    if sense == MAXIMIZE:
        best_idx = np.argsort(x)[-top_n:]
    elif sense == MINIMIZE:
        best_idx = np.argsort(x)[:top_n]
    else:
        raise ValueError(f"Invalid sense {sense} provided.")
    return best_idx
