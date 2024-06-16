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

import typing as tp

import numpy as np
import pandas as pd
from pydantic import BaseModel

import optimizer.types
from optimizer import Repair, StatefulOptimizationProblem, UserDefinedRepair

_TRange = tp.Tuple[float, float]


class _RepairViolation(BaseModel):
    """
    Private dataclass to store repair name and repair violations.
    Should not be used outside OptimizationExplanation.
    """
    name: str
    violations: tp.List[bool]


class _Point(BaseModel):
    """
    Dataclass to store point position in the 2D explanation space.
    """
    x_axis: float
    y_axis: float


class _Dependency(BaseModel):
    """
    Dataclass to store dependency between the
    explained_parameter and the dependent_function in the 2D explanation space.
    """
    dependent_function_name: str
    x_axis: tp.List[float]
    y_axis: tp.List[float]


class OptimizationExplanation(BaseModel):
    """
    Dataclass that stores the information about single optimization explanation.
    This class can be used to show optimization explanation
    in any visual form (e.g. table, plot).
    """
    explained_parameter: str
    problem_sense: tp.Literal["maximize", "minimize"]
    initial_point: _Point
    optimized_point: _Point
    dependency: _Dependency
    opt_bounds: _TRange
    repair_violations: tp.List[_RepairViolation]


def extract_dependent_function_value(
    dependent_function: optimizer.types.ReducesMatrixToSeries,
    state: pd.DataFrame,
    optimizable_parameter_value: float,
    optimizable_parameter: str,
) -> float:
    """
    Returns the value of the dependent function as a single float number.

    Handles the cases where the dependent function
    returns a ``pd.Series`` or a ``np.ndarray``, raises an error otherwise
    """
    state = state.copy()
    state[optimizable_parameter] = optimizable_parameter_value
    dependent_function_value = dependent_function(state)
    if isinstance(dependent_function_value, pd.Series):
        return float(dependent_function_value.iloc[0])
    elif isinstance(dependent_function_value, np.ndarray):
        return float(dependent_function_value[0])
    unexpected_dependent_function_return_type = type(dependent_function_value)
    raise ValueError(
        "Unexpected return type for `dependent_function`:"
        f" {unexpected_dependent_function_return_type}",
    )


def get_problem_objective(
    problem: StatefulOptimizationProblem,
    apply_penalties: bool,
    apply_repairs: bool,
) -> tp.Callable[[pd.DataFrame], pd.Series]:
    """Returns problem.objective"""
    return _ObjectiveWrapper(
        problem=problem,
        apply_repairs=apply_repairs,
        apply_penalties=apply_penalties,
    )


class _ObjectiveWrapper(object):
    def __init__(
        self,
        problem: optimizer.StatefulOptimizationProblem,
        apply_penalties: bool,
        apply_repairs: bool,
    ) -> None:
        self._problem = problem
        self._apply_repairs = apply_repairs
        self._apply_penalties = apply_penalties

    def __call__(self, state: pd.DataFrame) -> pd.Series:
        """
        Evaluates problem's objective with provided state instead of using problem's
        """
        if self._apply_repairs:
            state = self._problem.apply_repairs(state)
        objective = self._problem.objective(state)
        if self._apply_penalties:
            objective += self._problem.calculate_penalty(state)
        return objective

    def __repr__(self) -> str:
        if self._apply_repairs and self._apply_penalties:
            return "Objective with Penalties and Repairs"
        elif self._apply_penalties:
            return "Objective with Penalties"
        elif self._apply_repairs:
            return "Objective with Repairs"
        return "Objective"


def create_repair_violations(
    repairs: tp.List[Repair],
    state_with_grid: pd.DataFrame,
) -> tp.List[_RepairViolation]:
    user_defined_repairs = [
        repair
        for repair in repairs
        if isinstance(repair, UserDefinedRepair)
    ]
    repair_violations = []
    for repair in user_defined_repairs:
        repair(state_with_grid)
        repair_violations.append(
            _RepairViolation(
                name=repair.name,
                violations=repair.constraint.violated,
            ),
        )
    return repair_violations
