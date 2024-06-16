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
Wall time based stopper.
"""

import time
import typing as tp

from optimizer import OptimizationProblem
from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper


class WallTimeStopper(BaseStopper):
    """
    This class will return True from `stop` if N minutes have overlapped since the
    first call to `update`.
    """

    def __init__(self, minutes: float, **kwargs: tp.Any) -> None:
        """Constructor.

        Args:
            minutes: number of minutes to wait before returning True in `stop`.

        Raises:
            ValueError: if `minutes` is negative.
        """
        super().__init__(**kwargs)

        if minutes < 0:
            raise ValueError(
                f"Provided value for minutes must be nonnegative. Provided {minutes}"
            )

        self.seconds = minutes * 60
        self.first_call: tp.Optional[float] = None

    def stop(self) -> bool:
        """
        Stop getter.

        Returns:
            True if the search should stop.
            I.e., if the time passed since the first update
            exceeds the threshold (specified in the init).
            If called before `update`, returns `False`.
        """
        if self.first_call is None:
            return False
        time_since_first_call = time.perf_counter() - self.first_call
        return time_since_first_call >= self.seconds

    def update(
        self,
        solver: tp.Optional[Solver] = None,
        problem: tp.Optional[OptimizationProblem] = None,
    ) -> None:
        """Updates the internal stopping state.

        Args:
            solver: solver object used for implementing a various stopping criteria
            problem: problem object used for implementing various stopping criteria
        """
        if self.first_call is None:
            self.first_call = time.perf_counter()
