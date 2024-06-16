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
Maximum objective based stopper.
"""

import typing as tp

import numpy as np

from optimizer import OptimizationProblem
from optimizer.solvers.base import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.stoppers.utils import get_best
from optimizer.types import MINIMIZE


class NoImprovementStopper(BaseStopper):
    """
    This class will stop the search after N iterations without improvement in
    the best objective value.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        first_delta: float = 0.0,
        **kwargs: tp.Any,
    ) -> None:
        """Constructor.

        Args:
            patience: number of iterations without improvement.
            min_delta: minimum difference to be considered an improvement.

                - Zero means any improvement will be reset the patience counter.
                - Must be positive, any improvements will be compared
                  to it according to `solver.sense`

            first_delta: minimum difference between the first best solution to be
                considered an improvement.

                - Counting up toward `patience` will not begin until a solution that
                  is at least `first_delta` better than the first best solution.

        Raises:
            ValueError: if sense is not "minimize" or "maximize".
            ValueError: if min_delta is negative.
            ValueError: if first_delta is negative.
        """

        super().__init__(**kwargs)

        self.patience = _validate_argument_is_positive(patience, "patience")
        self.min_delta = _validate_argument_is_positive(min_delta, "min_delta")
        self.first_delta = _validate_argument_is_positive(first_delta, "first_delta")
        self.no_improvement = 0
        self.best: tp.Optional[float] = None
        self.first_best: tp.Optional[float] = None

    def _update(self, best: float, delta: float) -> None:
        """Internal update. Update counters based on improvements.

        Args:
            delta: float, delta between external best and internal best solution.
        """
        if delta > self.min_delta:
            # We found a better objective value.
            self.best = best
            self.no_improvement = 0
        else:
            self.no_improvement += 1

    def update(
        self,
        solver: tp.Optional[Solver] = None,
        problem: tp.Optional[OptimizationProblem] = None,
    ) -> None:
        """Update internal state based on best objective value seen so far.

        Args:
            solver: solver object;
                used to access the best objective and parameters
            problem: problem object; unused

        Raises:
            ValueError: if solver is None

        Returns:
            bool, True if the search should stop.
        """
        if solver is None:
            raise ValueError("Please provide a valid solver.")

        is_minimize = solver.sense == MINIMIZE

        # init best value for the first run
        if self.best is None:
            self.best = np.inf if is_minimize else -np.inf

        best, delta = get_best(solver.objective_values, self.best, is_minimize)

        if self.first_delta > 0.0:
            self.first_best = best if self.first_best is None else self.first_best
            _, delta_btw_first = get_best([best], self.first_best, is_minimize)

            # Only begin normal operation after we've found something better than the
            # first best solution by at least `self.first_delta`.
            if self.first_best != best and delta_btw_first > self.first_delta:
                self._update(best, delta)
        else:
            self._update(best, delta)

        self._stop = self.no_improvement >= self.patience


def _validate_argument_is_positive(value: float, argument_name: str) -> float:
    if value < 0:
        raise ValueError(f"{argument_name} must be positive. Provided {value}.")
    return value
