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
Holds definitions for the OptimizationProblem.
"""
import typing as tp

import numpy as np

from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.exceptions import InvalidObjectiveError
from optimizer.problem.user_defined_constraint import (
    UserDefinedPenalty,
    UserDefinedRepair,
)
from optimizer.types import (
    MAXIMIZE,
    MINIMIZE,
    MapsMatrixToMatrix,
    Matrix,
    Predictor,
    ReducesMatrixToSeries,
    TSense,
    Vector,
)
from optimizer.utils import check_matrix
from optimizer.utils.validation import wrap_with_reducer_dim_check

_TFirst = tp.TypeVar("_TFirst", bound=tp.Union[Penalty, Repair])
_TSecond = tp.TypeVar("_TSecond")
_TCallable = tp.Union[tp.Callable[[Matrix], Vector], tp.Callable[[Matrix], Matrix]]

_TUserDefinedPenalty = tp.Union[Penalty, UserDefinedPenalty]
_TUserDefinedRepair = tp.Union[Repair, UserDefinedRepair]
_TListUserDefinedPenalty = tp.List[_TUserDefinedPenalty]
_TListUserDefinedRepair = tp.List[_TUserDefinedRepair]

_UserDefinedPenalty = (Penalty, ReducesMatrixToSeries)
_UserDefinedRepair = (Repair, MapsMatrixToMatrix)


class OptimizationProblem:
    """
    Represents a basic optimization problem.
    Meaning all variables passed to __call__ are optimizable.
    """

    def __init__(
        self,
        objective: tp.Union[ReducesMatrixToSeries, Predictor],
        penalties: tp.Optional[tp.Union[
            Penalty,
            ReducesMatrixToSeries,
            tp.List[tp.Union[Penalty, ReducesMatrixToSeries]],
        ]] = None,
        repairs: tp.Optional[tp.Union[
            Repair,
            MapsMatrixToMatrix,
            tp.List[tp.Union[Repair, MapsMatrixToMatrix]],
        ]] = None,
        sense: TSense = MINIMIZE,
    ) -> None:
        """Constructor.

        Args:
            objective: callable object representing the objective function.
            penalties: optional Penalties or callable or list of
                Penalties and/or callables.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
                - Repairs are always applied before the objective and penalties are
                calculated.
            sense: 'minimize' or 'maximize', how to optimize the objective.

        Raises:
            InvalidObjectiveError: when objective is not callable or doesn't have a
                callable ``predict`` method.
            ValueError: if sense is not "minimize" or "maximize".
            ValueError: if penalties or repairs are invalid types/not callable.

        """
        # Beware all validations below, we don't check for the dimensions
        # (python 3.8 limitation). So checking for `ReducesMatrixToSeries` or
        # `MapsMatrixToMatrix` during the runtime is the same
        # as checking for `callable`.
        # This is why we wrap penalties and repairs with
        # `_reduce_with_dim_check` and `_map_with_dim_check`.
        self._objective = _validate_objective(objective)
        self.penalties: _TListUserDefinedPenalty = _validate_is_iterable_of_valid_type(
            penalties, _UserDefinedPenalty, "penalties",  # type: ignore
        )
        self.repairs: _TListUserDefinedRepair = _validate_is_iterable_of_valid_type(
            repairs, _UserDefinedRepair, "repairs",  # type: ignore
        )
        self.sense = _validate_sense(sense)

    def __call__(
        self,
        parameters: Matrix,
        apply_penalties: bool = True,
        apply_repairs: bool = True,
    ) -> tp.Tuple[Vector, Matrix]:
        """
        Evaluate the OptimizationProblem on a Matrix of parameters.

        Args:
            parameters: Matrix of parameters to evaluate.
            apply_penalties: if True, include penalties in objective,
                evaluate without applying penalties otherwise.
            apply_repairs: if True, apply repairs to parameters
                before evaluating objective,
                evaluate without applying repairs otherwise.

        Returns:
            Vector of objective values and (possibly repaired) parameter matrix.
        """
        check_matrix(parameters)

        if self.repairs and apply_repairs:
            parameters = self.apply_repairs(parameters)

        objectives = self._objective(parameters)

        if self.penalties and apply_penalties:
            objectives += self.calculate_penalty(parameters)

        return objectives, parameters

    @property
    def objective(self) -> ReducesMatrixToSeries:
        return self._objective

    def calculate_penalty(self, parameters: Matrix) -> Vector:
        """
        Calculates total penalty across all ``self.penalties``
        with ``self.sense`` taken into account
        (i.e., penalty is negative for ``maximize`` sense and positive for ``minimize``)

        Args:
            parameters: Matrix of parameters for calculating penalties.

        Returns:
            Vector of penalty values with correct sign.
        """
        check_matrix(parameters)

        total_penalty = np.zeros(parameters.shape[0])
        for penalty in self.penalties:
            total_penalty += penalty(parameters)

        if self.sense == MAXIMIZE:
            total_penalty *= -1

        return total_penalty

    def apply_repairs(self, parameters: Matrix) -> Vector:
        """Apply repairs.

        Args:
            parameters: Matrix of parameters to repair.

        Returns:
            Matrix.
        """
        check_matrix(parameters)
        repaired = parameters.copy()
        for repair in self.repairs:
            repaired = repair(repaired)
        return repaired


def _validate_objective(objective: tp.Any) -> ReducesMatrixToSeries:
    if isinstance(objective, Predictor):
        objective = objective.predict
    elif isinstance(objective, ReducesMatrixToSeries):
        pass
    else:
        raise InvalidObjectiveError(
            "Provided objective must be callable or have a .predict method."
        )
    return wrap_with_reducer_dim_check(objective)


def _validate_is_iterable_of_valid_type(
    value: tp.Any,
    expected_type: tp.Tuple[tp.Type[_TFirst], tp.Type[_TSecond]],
    validated_arg: str,
) -> tp.List[tp.Union[_TFirst, _TSecond]]:
    """
    Casts input ``value`` to list of expected_type
    Raises:
        ValueError:
            If any of input elements is not of ``expected_type``
        ValueError:
            If ``expected_type`` doesn't contain ReducesMatrixToSeries
            or MapsMatrixToMatrix type
    """
    if value is None:
        return []
    multiple_values = [value] if not isinstance(value, list) else value.copy()
    _check_is_expected_type(multiple_values, validated_arg, expected_type)
    return _wrap_callables_with_dim_checks(
        multiple_values,
        expected_type
    )


def _wrap_callables_with_dim_checks(
    callables: tp.List[tp.Union[MapsMatrixToMatrix, ReducesMatrixToSeries]],
    expected_type: tp.Tuple[tp.Type[_TFirst], tp.Type[_TSecond]],
) -> tp.List[tp.Union[_TFirst, _TSecond]]:
    if expected_type == _UserDefinedPenalty:
        return [
            arg if isinstance(arg, Penalty)  # type: ignore
            else UserDefinedPenalty(arg)
            for arg in callables
        ]
    elif expected_type == _UserDefinedRepair:
        return [
            arg if isinstance(arg, Repair)  # type: ignore
            else UserDefinedRepair(arg)
            for arg in callables
        ]
    raise ValueError(
        f"`expected_type` must either be `{_UserDefinedPenalty}` "
        f"or `{_UserDefinedRepair}`",
    )


def _check_is_expected_type(
    multiple_args: tp.List[tp.Any],
    arg_name: str,
    expected_types: tp.Tuple[tp.Type[tp.Any], ...],
) -> None:
    invalid_objects = [
        single_arg
        for single_arg in multiple_args
        if not isinstance(single_arg, expected_types)
    ]
    if invalid_objects:
        invalid_object_names = [
            type(invalid_obj).__name__ for invalid_obj in invalid_objects
        ]
        expected_type_repr = " | ".join(t.__name__ for t in expected_types)
        raise ValueError(
            f"Found {invalid_object_names} provided as a {arg_name}, "
            f"but a {expected_type_repr} is required."
        )


def _validate_sense(sense: TSense) -> TSense:
    if sense not in {MAXIMIZE, MINIMIZE}:
        raise ValueError(f"{sense} is an invalid optimization sense.")
    return sense
