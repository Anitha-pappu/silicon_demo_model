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
Abstract definitions for domain classes.
"""

import abc
import typing as tp

import numpy as np
import numpy.typing as npt
from sklearn.utils import check_random_state

from optimizer.types import TCategorical, TReal, Vector

TDimension = tp.Union[
    tp.Tuple[TReal, TReal],
    tp.Set[TCategorical],
    "BaseDimension",
]


class BaseDimension(abc.ABC):
    """
    Base class for specifying the dimension of an optimization problem.

    A simplified version of the skopt API here:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py
    """

    def sample(
        self,
        n_samples: int = 1,
        random_state: tp.Optional[tp.Union[int, np.random.RandomState]] = None,
    ) -> Vector:
        """Draw a random sample from the dimension.

        Args:
            n_samples: optional int, number of samples returned.
            random_state: optional int or np.random.RandomSate, sets the random seed of
                the sample operation.

        Returns:
            np.ndarray of samples.
        """
        rng = check_random_state(random_state)
        sample = self._sample(n_samples=n_samples, random_state=rng)
        return sample

    @abc.abstractmethod
    def _sample(
        self, n_samples: int, random_state: np.random.RandomState,
    ) -> Vector:
        """Draw a random sample from the dimension.

        Args:
            n_samples: int, number of samples returned.
            random_state: np.random.RandomSate, seeded rng to do sampling with.

        Returns:
            np.ndarray of samples.
        """

    @property
    @abc.abstractmethod
    def bounds(self) -> tp.Tuple[tp.Any, ...]:
        """Get the bounds of the dimension.

        Returns:
            Tuple of elements.
        """

    def __len__(self) -> int:
        return len(self.bounds)
