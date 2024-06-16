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
Constraint utility functions.
"""

import typing as tp

import numpy as np

from optimizer.types import Matrix, Vector


class ConstantCallable:
    """
    Wrapper class to wrap a constant object.
    """

    def __init__(self, const: float):
        """Constructor.

        Args:
            const: float.
        """
        self.const: float = const
        self.output: tp.Optional[Vector] = None

    def __call__(self, parameters: Matrix) -> Vector:
        """Return the provided constant n times.

        Args:
            parameters: n x d matrix.

        Returns:
            Vector with n entries of the provided constant.
        """
        if self.output is None or len(self.output) != parameters.shape[0]:
            self.output = self.const * np.ones(parameters.shape[0], dtype=float)

        return self.output
