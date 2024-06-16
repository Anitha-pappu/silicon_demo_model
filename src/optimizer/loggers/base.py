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
Logger base module.
"""

import abc
import typing as tp

from optimizer import OptimizationProblem
from optimizer.solvers import Solver


class LoggerMixin(abc.ABC):
    """
    Logging interface mixin.
    """

    @abc.abstractmethod
    def log(
        self,
        *,
        problem: tp.Optional[OptimizationProblem] = None,
        solver: tp.Optional[Solver] = None,
    ) -> None:
        """
        Implement this method. It should log something i.e.,
        update logger state with the relative information

        Args:
            problem: evaluated problem
            solver: a solver used for evaluating the ``problem``
        """

    @abc.abstractmethod
    def clone(self) -> 'LoggerMixin':
        """
        Creates a clone of the logger WITHOUT any state copying
        from the original instance
        """

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(...)"
