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
Penalty logger package.
"""

import typing as tp

from optimizer.constraint import Penalty
from optimizer.constraint.constraint import InequalityConstraint
from optimizer.loggers.base import LoggerMixin
from optimizer.problem import OptimizationProblem
from optimizer.problem.user_defined_constraint import UserDefinedPenalty
from optimizer.solvers import Solver


class TPenaltyLogRecord(tp.TypedDict):
    value: tp.List[float]
    penalty: tp.List[float]
    lower_bound: tp.Optional[tp.List[float]]
    upper_bound: tp.Optional[tp.List[float]]


class PenaltyLogger(LoggerMixin):
    def __init__(self) -> None:
        """
        Logs the information about penalties state at each iteration.
        See more details in ``log`` method
        """
        self.log_records: tp.List[tp.Any] = []

    def clone(self) -> 'PenaltyLogger':
        return PenaltyLogger()

    def log(
        self,
        *,
        problem: tp.Optional[OptimizationProblem] = None,
        solver: tp.Optional[Solver] = None,  # pylint: disable=unused-argument
    ) -> None:
        """
        Log the following information about penalties:
            * "lower_bound" and "upper_bound"
                for each instance of ``InequalityConstraint``
            * "value" and "penalty" for all types of penalties

        Args:
            problem: problem with penalties
            solver: is not used

        Raises:
            ValueError: if ``problem`` is not provided
            ValueError: when any of ``problem.penalties`` has not been evaluated
        """

        if problem is None:
            raise ValueError("Please provide a problem")

        penalties: tp.List[tp.Union[Penalty, UserDefinedPenalty]] = problem.penalties

        if isinstance(penalties, Penalty):
            penalties = [penalties]

        by_penalty_log_record: tp.Dict[str, TPenaltyLogRecord] = {}

        for i, penalty in enumerate(penalties):

            if penalty.calculated_penalty is None:
                raise ValueError("Attempted to log an unevaluated constraint.")

            name = f"penalty_{i}" if penalty.name is None else penalty.name
            constraint = penalty.constraint  # type: ignore

            penalty_data = {
                "value": list(constraint.constraint_values),
                "penalty": list(penalty.calculated_penalty),
                "lower_bound": None,
                "upper_bound": None,
            }

            if isinstance(constraint, InequalityConstraint):
                # If upper and lower bounds are some kind of function,
                # their values will be useful and should be logged too.
                # Otherwise, just log the single constant.
                penalty_data["lower_bound"] = (
                    constraint.lower_bound_eval
                    if callable(constraint.lower_bound)
                    else constraint.lower_bound
                )
                penalty_data["upper_bound"] = (
                    constraint.upper_bound_eval
                    if callable(constraint.upper_bound)
                    else constraint.upper_bound
                )
            by_penalty_log_record[name] = tp.cast(TPenaltyLogRecord, penalty_data)

        self.log_records.append(by_penalty_log_record)
