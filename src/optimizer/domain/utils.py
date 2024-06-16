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
Domain utils module.
"""

import typing as tp

import numpy as np

from optimizer.domain import (
    CategoricalDimension,
    IntegerDimension,
    RealDimension,
)
from optimizer.domain.base import BaseDimension
from optimizer.types import TBounds, TReal


def check_continuous_domain(
    domain: tp.Sequence[tp.Union[TBounds, RealDimension]]
) -> tp.List[TBounds]:
    """Confirm a continuous domain specification or error.

    Args:
        domain: list of tuples and/or RealDimensions, upper and lower bounds for
            each dimension or a RealDimension specification.

    Raises:
        ValueError if an invalid domain specification is provided.

    Returns:
        List of upper and lower bounds for each dimension.
    """
    out = []

    for i, dimension in enumerate(domain):
        if isinstance(dimension, RealDimension):
            dimension = dimension.bounds

        elif isinstance(dimension, BaseDimension):
            raise ValueError(
                f"Invalid BaseDimension type {type(dimension)} at index {i}."
            )

        if len(dimension) == 2:
            out.append(dimension)

        else:
            raise ValueError(
                f"Invalid dimension {dimension} at index {i} with type "
                f"{type(dimension)}. Must be Tuple[Real, Real] or RealDimension."
            )

    try:
        limits = np.array(out, dtype="float")
    except ValueError:
        raise ValueError("Domain must be all numeric types.")

    if not np.all(np.isfinite(limits)):
        raise ValueError("Domain entries all must be finite.")

    return out


def check_discrete_domain(
    domain: tp.List[tp.Union[tp.List[tp.Any], IntegerDimension, CategoricalDimension]]
) -> tp.List[tp.List[tp.Any]]:
    """Confirm a discrete domain specification or error.

    Args:
        domain: list of lists, IntegerDimension, or CategoricalDimension, specifying the
            possible choices for each dimension in the domain.

    Raises:
        ValueError if an invalid domain specification is provided.

    Returns:
        List of lists specifying the choices for each dimension.
    """
    out = []
    dim: tp.List[tp.Any] = []

    for i, dimension in enumerate(domain):
        if isinstance(dimension, CategoricalDimension):
            dim = list(dimension.bounds)  # Get the bounds as a tuple.

        elif isinstance(dimension, IntegerDimension):
            dim = list(range(dimension.bounds[0], dimension.bounds[1] + 1))
        elif isinstance(dimension, BaseDimension):
            raise ValueError(
                f"Invalid BaseDimension type {type(dimension)} at index {i}."
            )
        else:
            dim = dimension

        if len(dim) == 0:
            raise ValueError(
                f"Dimension {i} is empty. Each "
                "dimension must have at least one value."
            )

        out.append(dim)

    return out