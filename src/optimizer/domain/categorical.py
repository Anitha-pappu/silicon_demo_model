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
Categorical dimension module.
"""

import typing as tp

import numpy as np

from optimizer.domain.base import BaseDimension
from optimizer.types import TCategorical, TReal, Vector


class CategoricalDimension(BaseDimension):
    """
    Class for representing a dimension of categorical choices.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py
    """

    def __init__(self, categories: tp.Iterable[TCategorical]):
        """Constructor

        Args:
            categories:
        """
        self.categories: tp.Tuple[TCategorical, ...] = tuple(categories)
        self.categories_array = np.array(categories, dtype="O")

    def _sample(self, n_samples: int, random_state: np.random.RandomState) -> Vector:
        """Draw a random sample from the dimension.

        Args:
            n_samples: int, number of samples returned.
            random_state: np.random.RandomSate, seeded rng to do sampling with.

        Returns:
            np.ndarray of samples.
        """
        return random_state.choice(self.categories_array, size=n_samples, replace=True)

    def __getitem__(self, item: tp.List[int]) -> TCategorical:
        """Allow for array indexing into categories for selecting an array of items.

        Example::

            >>> dim = CategoricalDimension(["hello", "world"])
            >>> dim[[0, 1, 1]]

        array(['hello', 'world', 'world'], dtype=object)

        Args:
            item:
        """
        return tp.cast(TCategorical, self.categories_array[item])

    @property
    def bounds(self) -> tp.Tuple[TCategorical, ...]:
        """Get all the categories of this dimension.

        Returns:
            Tuple of elements.
        """
        return self.categories
