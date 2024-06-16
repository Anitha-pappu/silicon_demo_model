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
Basic logger module.
"""

import typing as tp

import numpy as np
import pandas as pd

from optimizer import OptimizationProblem
from optimizer.loggers.base import LoggerMixin
from optimizer.solvers import Solver
from optimizer.types import MINIMIZE, TSense


class TLogRecord(tp.TypedDict):
    mean_objective: float
    std_objective: float
    median_objective: float
    first_quartile_objective: float
    third_quartile_objective: float
    min_objective: float
    max_objective: float


# Map the aggregate functions to their corresponding TypedDict keys in TLogRecord
# This is a workaround for mypy to avoid using f-strings
# (which are compiled in runtime) as a TypedDict key
_agg_function_map: tp.Dict[
    str,
    tp.Union[tp.Literal["min_objective"], tp.Literal["max_objective"]]
] = {
    "min": "min_objective",
    "max": "max_objective",
}


class BasicLogger(LoggerMixin):
    def __init__(self) -> None:
        """
        Logs solver's summary statistics.

        Examples:
            Using the BasicLogger::

                >>> logger = BasicLogger()
                >>> parameters = solver.ask()
                >>> objectives, repaired_parameters = problem(parameters)
                >>> solver.tell(repaired_parameters, objectives)
                >>> logger.log(solver=solver)
        """
        self.sense: tp.Optional[TSense] = None
        self.log_records: tp.List[TLogRecord] = []

    def clone(self) -> 'BasicLogger':
        return BasicLogger()

    def log(
        self,
        *,
        problem: tp.Optional[OptimizationProblem] = None,  # pylint: disable=unused-argument
        solver: tp.Optional[Solver] = None,
    ) -> None:
        """
        Logs solver's objective values' statistics
        (mean, std, and five quartiles, i.e. min, max included) at the last iteration.
        Refer to ``TLogRecord`` to learn the output dict structure.

        Args:
            problem: is not used
            solver: used for solving a problem

        Raises:
            ValueError: if ``solver`` is not provided
        """

        if solver is None:
            raise ValueError("Please provide a solver")

        if self.sense is None:
            self.sense = solver.sense
        if self.sense != solver.sense:
            raise ValueError(
                f"Logger that originally logged {self.sense = } got solver "
                f"with sense {solver.sense = }.",
            )

        objective_values = solver.objective_values

        min_, q25, med, q75, max_ = np.quantile(
            objective_values, [0, 0.25, 0.5, 0.75, 1]
        )
        self.log_records.append(
            {
                "mean_objective": float(np.mean(objective_values)),
                "std_objective": float(np.std(objective_values)),
                "median_objective": med,
                "first_quartile_objective": q25,
                "third_quartile_objective": q75,
                "min_objective": min_,
                "max_objective": max_,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Converts stored log records values to a DataFrame"""
        df = pd.DataFrame(self.log_records)
        df.index.name = "iteration"

        df["overall_min_objective"] = df["min_objective"].expanding().min()
        df["overall_max_objective"] = df["max_objective"].expanding().max()

        return df

    @property
    def first_quartile_objective(self) -> pd.Series:
        return pd.Series(
            [record["first_quartile_objective"] for record in self.log_records],
            dtype=float,
        )

    @property
    def third_quartile_objective(self) -> pd.Series:
        return pd.Series(
            [record["third_quartile_objective"] for record in self.log_records],
            dtype=float,
        )

    @property
    def median_objective(self) -> pd.Series:
        return pd.Series(
            [record["median_objective"] for record in self.log_records],
            dtype=float,
        )

    @property
    def best_seen_objective(self) -> pd.Series:
        if self.sense is None:
            raise ValueError("Please call this logger's `log` method at least once")
        agg_function = "min" if self.sense == MINIMIZE else "max"
        log_record_key = _agg_function_map[agg_function]

        return pd.Series(
            [record[log_record_key] for record in self.log_records],
            dtype=float,
        ).expanding().aggregate(agg_function)
