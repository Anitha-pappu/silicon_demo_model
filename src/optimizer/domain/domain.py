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
Domain module.
"""

import typing as tp

import numpy as np
from sklearn.utils import check_random_state

from optimizer.domain.base import BaseDimension, TDimension
from optimizer.domain.categorical import CategoricalDimension
from optimizer.domain.integer import IntegerDimension
from optimizer.domain.real import RealDimension
from optimizer.types import Vector

TDomain = tp.Union["Domain", tp.List[TDimension]]


def check_dimension(
    dimension: TDimension
) -> BaseDimension:
    """Convert the provided dimension specification to a BaseDimension object.

    Also does error checking on `dimension` to be sure that it is a supported type.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py

    Args:
        dimension: dimension specification. Can be:
            - Tuple of two floats for a Real dimension.
            - Tuple of two ints for an Integer dimension.
            - Set of strings for a Categorical domain.
            - An object extending BaseDimension (Real, Integer, Categorical).

    Raises:
        ValueError if the dimension provided isn't one of the above specifications.

    Returns:
        BaseDimension object.
    """
    _validate_dimension(dimension)

    if isinstance(dimension, BaseDimension):
        return dimension

    if isinstance(dimension, tuple):
        lower_bound, upper_bound = dimension

        if isinstance(lower_bound, int) and isinstance(upper_bound, int):
            return IntegerDimension(lower_bound, upper_bound)

        elif isinstance(lower_bound, float) or isinstance(upper_bound, float):
            return RealDimension(lower_bound, upper_bound)

    if isinstance(dimension, set):
        return CategoricalDimension(sorted(dimension))

    raise ValueError(
        f"Provided dimension with type `{type(dimension)}` must be either "
        "a tuple[Real, Real] for Real dimension, or "
        "a tuple[int, int] for Integer dimension, or "
        "a set[str | float | int | bool] for CategoricalDimension, or "
        "an object extending BaseDimension."
    )


def _validate_dimension(dimension: TDimension) -> None:
    if not isinstance(dimension, (tuple, set, BaseDimension)):
        raise ValueError(
            f"Invalid dimension {dimension}. "
            f"Provided dimension with type `{type(dimension)}` "
            "must be list, tuple, BaseDimension"
        )


class Domain(tp.Sequence[BaseDimension]):
    """
    Class for handling mixed domain problems.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py
    """

    def __init__(self, dimensions: tp.List[TDimension]):
        """Constructor.

        Args:
            dimensions: List. Each dimension can be:
                * `(low, high)` tuple for a Real or Integer dimension (based on type).
                * `(low, high, "categorical")` to specify a real valued categorical.
                * List of objects for a Categorical domain.
                * An object extending BaseDimension (Real, Integer, Categorical).
        """
        self.dimensions = [check_dimension(d) for d in dimensions]

    def __len__(self) -> int:
        """Get the number of dimensions.

        Returns:
            int.
        """
        return len(self.dimensions)

    @tp.overload
    def __getitem__(self, index: int) -> BaseDimension:
        ...

    @tp.overload
    def __getitem__(self, s: slice) -> tp.Sequence[BaseDimension]:
        ...

    def __getitem__(
            self,
            index: tp.Union[int, slice],
    ) -> tp.Union[BaseDimension, tp.Sequence[BaseDimension]]:
        """Get the dimension(s) at the provided index or slice.

        Args:
            index: Union[int, slice], index or slice.

        Returns:
            BaseDimension (indexing) or Sequence of dimensions (slicing).
        """
        if isinstance(index, slice):
            return self.dimensions[index.start:index.stop:index.step]

        return self.dimensions[index]

    def sample(
        self,
        n_samples: int = 1,
        random_state: tp.Optional[tp.Union[int, np.random.RandomState]] = None,
    ) -> Vector:
        """Randomly sample the domain.

        Args:
            n_samples: optional int, number of samples returned.
            random_state: optional int or np.random.RandomSate, sets the random seed of
                the sample operation.

        Returns:
            np.ndarray of sampled points with dimension (n_samples, len(self))
        """
        rng = check_random_state(random_state)

        columns = []

        for d in self.dimensions:
            columns.append(d.sample(n_samples=n_samples, random_state=rng))

        return np.column_stack(columns)
