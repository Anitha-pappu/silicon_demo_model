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
Best solution logger module.
"""

import typing as tp

import numpy as np
import pandas as pd

from optimizer import OptimizationProblem
from optimizer.loggers.base import LoggerMixin
from optimizer.loggers.penalty import TPenaltyLogRecord
from optimizer.solvers import Solver
from optimizer.types import Vector

_TLogRecords = tp.List[tp.Dict[str, tp.Union[TPenaltyLogRecord, Vector, float]]]


class BestTrajectoryLogger(LoggerMixin):
    """
    Logs the best solution and objective value for each iteration

    Examples:
        Using the logger::

            >>> logger = BestTrajectoryLogger()
            >>> parameters = solver.ask()
            >>> objectives, repaired_parameters = problem(parameters)
            >>> solver.tell(repaired_parameters, objectives)
            >>> logger.log(solver=solver)
    """

    def __init__(self) -> None:
        """Constructor.
        """
        self.log_records: _TLogRecords = []

    def clone(self) -> 'BestTrajectoryLogger':
        return BestTrajectoryLogger()

    def log(
        self,
        *,
        problem: tp.Optional[OptimizationProblem] = None,
        # pylint: disable=unused-argument
        solver: tp.Optional[Solver] = None,
    ) -> None:
        """
        Log ``solver``'s best solution and corresponding objective
        for the given iteration.

        Args:
            problem: is not used
            solver: used for solving a problem

        Raises:
            ValueError: if ``solver`` is not provided
        """

        if solver is None:
            raise ValueError("Please provide a solver")

        solution, objective_value = solver.best()

        self.log_records.append(
            {"solution": solution, "objective_value": objective_value},
        )

    def to_frame(self) -> pd.DataFrame:
        """Converts stored log records values to a DataFrame"""
        df = pd.DataFrame(self.log_records)
        df.index.name = "iteration"
        return df

    @property
    def objective_values(self) -> Vector:
        return self.to_frame()["objective_value"]

    @property
    def solutions(self) -> Vector:
        return np.stack(self.to_frame()["solution"])
