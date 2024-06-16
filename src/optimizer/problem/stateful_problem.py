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
Holds definitions for the StatefulOptimizationProblem
"""

import typing as tp
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from optimizer import types
from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.problem.problem import OptimizationProblem
from optimizer.types import (
    MINIMIZE,
    MapsMatrixToMatrix,
    Matrix,
    Predictor,
    ReducesMatrixToSeries,
    TColumn,
    TSense,
    Vector,
)
from optimizer.utils.functional import safe_subset
from optimizer.utils.validation import check_matrix

from ._internal_state import (
    check_internal_state_needs_update,
    create_internal_state,
)

_TState = tp.Union[Vector, Matrix]


class StatefulOptimizationProblem(OptimizationProblem):
    """
    Optimization problem representing the case where both optimizable and
    non-optimizable parameters must be handled.
    """

    def __init__(
        self,
        objective: tp.Union[ReducesMatrixToSeries, Predictor],
        state: _TState,
        optimizable_columns: tp.Iterable[TColumn],
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
    ):
        """Constructor.

        Args:
            objective: callable object representing the objective function.
            state: Vector representing the current state of the problem.
            optimizable_columns: iterable of columns/dimensions that can be optimized.
            penalties: optional Penalty or list of Penalties.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
            sense: 'minimize' or 'maximize', how to optimize the objective.
        """
        super().__init__(
            objective=objective,
            penalties=penalties,
            repairs=repairs,
            sense=sense,
        )
        self.optimizable_columns = list(optimizable_columns)
        # An internal current state will save us time for any processing
        # we might need to apply to the current state before subbing in
        # the optimizable variables in the __call__ method.
        self._internal_state: tp.Optional[Matrix] = None
        self._state: tp.Optional[types.Matrix] = None
        self.state = state  # assigning value to the _state through the setter

    @property
    def state(self) -> Matrix:
        """
        Returns the current (actual) state

        Returns:
            single-row 2D vector or pd.Dataframe
            representing the current state of the problem
        """
        return tp.cast(Matrix, self._state).copy()

    @state.setter
    def state(self, new_state: tp.Union[Matrix, Vector]) -> None:
        """
        Set actual state and reset internal state. The new state can be
        one of three things:
            - a single-row pd.DataFrame
            - a pd.Series with a numeric dtype*
            - a np.ndarray
        *pandas series with other dtypes are not allowed to protect users from
        potential dtype errors that arise when a dataframe's row is converted
        into a series.

        Args:
            new_state: vector representing the new state of the problem

        Raises:
            TypeError: in case of wrong input type
            KeyError: if any of the provided columns are not found
                in the problem's ``optimizable_columns``
        """
        validated_state = _validate_state(deepcopy(new_state))
        _validate_all_columns_are_mapped_one_to_one(
            validated_state,
            self.optimizable_columns,
            # to avoid state columns validation for first assignment
            self.non_optimizable_columns if self._state is not None else None,
        )
        self._state = validated_state

    @property
    def non_optimizable_columns(self) -> tp.List[TColumn]:
        """
        Get a list of columns that are not being optimized.

        Returns:
            List of ints or strings.
        """
        columns = (
            range(self._state.shape[1])
            if isinstance(self._state, np.ndarray)
            else tp.cast(pd.DataFrame, self._state).columns
        )
        return [
            column
            for column in columns
            if column not in set(self.optimizable_columns)
        ]

    def __call__(
        self,
        parameters: Matrix,
        apply_penalties: bool = True,
        apply_repairs: bool = True,
    ) -> tp.Tuple[Vector, Matrix]:
        """Evaluate the OptimizationProblem on a Matrix of parameters.

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
        params_plus_state = self.substitute_parameters(parameters)

        objectives, params_plus_state = super().__call__(
            params_plus_state, apply_penalties, apply_repairs,
        )

        return objectives, safe_subset(params_plus_state, self.optimizable_columns)

    def substitute_parameters(self, parameters: Matrix) -> Matrix:
        """Substitute the optimizable parameters into the current state.

        Args:
            parameters: Matrix of parameters to substitute into the current state.

        Returns:
            Matrix of parameters with non-optimizable parameters appended
            in correct order.
        """
        if not self.optimizable_columns:
            return self.state.copy()
        check_matrix(parameters)
        self._check_substitutable(parameters)
        substitute = (
            self._substitute_numpy
            if isinstance(self._state, np.ndarray)
            else self._substitute_pandas
        )
        return substitute(parameters)

    def _check_substitutable(self, parameters: Matrix) -> None:
        """
        Check that the column indexing will work when we try to substitute.

        Args:
            parameters: Matrix of parameters to substitute into the current state.

        Raises:
            `TypeError`: if there is a type mismatch between the state and parameters.
            `ValueError`: if `parameters` has too many columns.
            `KeyError`: if the `parameters.columns` doesn't match `self.state.columns`
                in case state is instance of `pd.DataFrame`.
        """
        if len(self.optimizable_columns) != parameters.shape[-1]:
            raise ValueError(
                f"`parameters` matrix has {parameters.shape[-1]} columns and problem "
                f"has {len(self.optimizable_columns)} optimizable columns. "
                "Lengths must match."
            )

        if isinstance(parameters, pd.DataFrame):
            missing_columns = (
                set(self.optimizable_columns).difference(parameters.columns)
            )
            if missing_columns:
                raise KeyError(
                    f"Optimizable columns {missing_columns} are not in the provided "
                    f"DataFrame with columns {parameters.columns}"
                )

    def _substitute_numpy(self, parameters: Matrix) -> Vector:
        """
        Substitute the optimizable parameters into the current state for a numpy
        array.

        Args:
            parameters: numpy array of parameters to substitute into the current state.

        Returns:
            A new numpy array with controls from ``self.state`` and optmimizable values
            from ``parameters``
        """
        if isinstance(parameters, pd.DataFrame):
            raise ValueError(
                "Provided `parameters` type (`pd.DataFrame`) is incompatible "
                "with problem's state type (`np.ndarray`). You can either switch to "
                "`pd.DataFrame` state (preferred) in problem or pass "
                "`np.ndarray` instance as parameters.",
            )

        self._update_internal_state_if_parameters_format_is_new(parameters)

        substitution = self._internal_state.copy()  # type: ignore
        substitution[:, self.optimizable_columns] = parameters
        return substitution

    def _substitute_pandas(self, parameters: Matrix) -> pd.DataFrame:
        """Substitute the optimizable parameters into the current state for a numpy
        array.

        Args:
            parameters: numpy array of parameters to substitute into the current state.

        Returns:
            A new DataFrame with non-optimizable columns values from
            ``self.state`` and controlled columns values from ``parameters``
        """
        self._update_internal_state_if_parameters_format_is_new(parameters)

        if isinstance(self._internal_state, (np.ndarray, np.matrix)):
            self._internal_state = pd.DataFrame(self._internal_state)

        substitution = self._internal_state.copy(deep=True)
        substitution[self.optimizable_columns] = parameters
        return substitution

    def _update_internal_state_if_parameters_format_is_new(
        self, parameters: Matrix,
    ) -> None:
        needs_update = check_internal_state_needs_update(
            parameters, self._internal_state, self._state,
        )
        if not needs_update:
            return
        self._internal_state = create_internal_state(
            parameters, self._state, self.non_optimizable_columns,
        )


def _validate_state(new_state: _TState) -> Matrix:
    if isinstance(new_state, pd.Series):
        if not is_numeric_dtype(new_state):
            raise _get_incorrect_state_type_error(
                f"series with non-numeric dtype: {new_state.dtype = }",
            )
        new_state = new_state.to_frame().T
    elif isinstance(new_state, pd.DataFrame):
        if not len(new_state) == 1:
            raise _get_incorrect_state_type_error(
                f"a dataframe with more than one row: {len(new_state) = }",
            )
    elif isinstance(new_state, np.ndarray):
        if new_state.ndim == 0:
            raise _get_incorrect_state_type_error(
                "a scalar; please reshape it to 1D/2D array",
            )
        if new_state.squeeze().ndim > 2:
            raise _get_incorrect_state_type_error(
                f"an n-dim array: {new_state.shape = }; "
                f"please reshape it to 1D/2D array",
            )
        new_state = new_state.reshape(1, -1)
    else:
        raise _get_incorrect_state_type_error(
            f"an object of unexpected type: {type(new_state) = }",
        )
    return new_state


def _validate_all_columns_are_mapped_one_to_one(
    state: Matrix,
    optimizable_columns: tp.List[TColumn],
    non_optimizable_columns: tp.Optional[tp.List[TColumn]],
) -> None:
    all_columns = list(
        state.columns
        if isinstance(state, pd.DataFrame)
        else np.arange(state.shape[1])
    )

    missing_opt_columns = set(optimizable_columns).difference(all_columns)
    if missing_opt_columns:
        raise KeyError(
            f"Not all required optimizable columns are provided in `new_state`, "
            f"missing columns: {sorted(list(missing_opt_columns))}."
        )

    extra_non_opt_columns = (
        set(all_columns).difference(non_optimizable_columns)
        if non_optimizable_columns is not None
        else None
    )
    if extra_non_opt_columns:
        warnings.warn(
            f"New state contains following extra non-optimizable columns: "
            f"{sorted(list(extra_non_opt_columns))}."
        )


def _get_incorrect_state_type_error(type_got: str) -> TypeError:
    return TypeError(
        f"`state` must be a single-row dataframe/"
        f"a numeric series/a numpy array. "
        f"Got {type_got}."
    )
