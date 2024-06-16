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
Holds definitions for the StatefulContextualOptimizationProblem
"""

import typing as tp
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from optimizer.constraint.penalty import Penalty
from optimizer.constraint.repair import Repair
from optimizer.problem.problem import OptimizationProblem
from optimizer.types import (
    MINIMIZE,
    MapsMatrixToMatrix,
    Matrix,
    Predictor,
    ReducesMatrixToSeries,
    TSense,
    Vector,
)
from optimizer.utils.functional import safe_subset
from optimizer.utils.validation import check_matrix

_T = tp.TypeVar("_T")


class StatefulContextualOptimizationProblem(OptimizationProblem):
    """
    Optimization problem representing the case where both optimizable and
    non-optimizable parameters must be handled.
    """

    def __init__(
        self,
        objective: tp.Union[ReducesMatrixToSeries, Predictor],
        state: tp.Union[Vector, Matrix],
        context_data: Matrix,
        optimizable_columns: tp.Sequence[tp.Union[str, int]],
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
        n_opt_blocks: int = 1,
    ) -> None:
        """
        Create a new instance of a Stateful Contextual Problem.

        Args:
            objective: callable object representing the objective function;
                this objective will be provided a dataset which also contains prepended
                context for each row; so if input at this call is (3 x 5) matrix
                and context was set to (7 x 4) then objective will receive
                (24 x 9) input matrix where each of 3 rows gets 7 additional
                context rows
            state: Vector representing the current state of the problem.
            optimizable_columns: list of columns/dimensions that can be optimized.
            penalties: optional Penalty or list of Penalties.
            repairs: optional Repairs or callable or list of Repairs and/or callables.
            sense: 'minimize' or 'maximize', how to optimize the objective.
            n_opt_blocks: Number of timestamps to replace controls for during the call:
                * 1 replaces only recent input;
                * N>1 replaces N most-recent timestamps going back through the context
                  i.e., N-latest controls will have the same control value
                  coming from parameters
        """
        super().__init__(
            objective=objective,
            penalties=penalties,
            repairs=repairs,
            sense=sense,
        )
        self.optimizable_columns = optimizable_columns
        self.context_data = context_data
        self.state = state
        context_size = context_data.shape[0]
        max_possible_n_opt_blocks = context_size + 1
        if n_opt_blocks > max_possible_n_opt_blocks:
            n_opt_blocks = max_possible_n_opt_blocks
            warnings.warn(
                f"`n_opt_blocks` can't be greater "
                f"than context length ({context_size}) + 1 = {n_opt_blocks}. "
                f"Using largest possible `n_opt_blocks={n_opt_blocks}`"
            )
        self.n_opt_blocks = n_opt_blocks
        self._internal_state: tp.Optional[tp.Union[Vector, Matrix]] = None

    @property
    def optimizable_columns(self) -> tp.Sequence[tp.Union[str, int]]:
        """
        Optimizable columns' getter.

        Returns:
            List or pd.Series.
        """
        return self._optimizable_columns

    @optimizable_columns.setter
    def optimizable_columns(self, new_columns: tp.Iterable[str]) -> None:
        """
        Optimizable columns' setter.
        Resets internal state.

        Args:
            new_columns: list of columns/dimensions that can be optimized.
        """
        self._optimizable_columns = list(new_columns)
        self._internal_state = None  # Reset internal state.

    @property
    def state(self) -> Vector:
        """
        Returns the current (actual) state

        Returns:
            vector representing the current state of the problem
        """
        return self._actual_state

    @state.setter
    def state(self, new_state: tp.Union[Matrix, Vector]) -> None:
        """
        Set actual state and reset internal state. The new state can be
        one of three things:
            - a single-row dataframe
            - a pandas series with a numeric dtype
            - a numpy array
        pandas series with other dtypes are not allowed to protect users from
        potential dtype errors that arrise when a dataframe's row is converted
        into a series.

        Args:
            new_state: vector representing the new state of the problem

        Raises:
            TypeError: in case of a wrong input type
        """
        new_state = deepcopy(new_state)

        type_error_msg = (
            "`state` must be a single-row dataframe, "
            "a numeric series, or a numpy array. "
            "Got {}."
        )

        if isinstance(new_state, pd.Series):
            if not is_numeric_dtype(new_state):
                raise TypeError(
                    type_error_msg.format(f"a series of type {new_state.dtype}")
                )
            new_state_frame = new_state.to_frame().T
            all_columns = new_state_frame.columns.tolist()
        elif isinstance(new_state, pd.DataFrame):
            if not len(new_state) == 1:
                raise TypeError(
                    type_error_msg.format(f"a dataframe with {len(new_state)} rows")
                )
            all_columns = new_state.columns.tolist()
        elif isinstance(new_state, np.ndarray):
            all_columns = list(np.arange(len(new_state)))
        else:
            raise TypeError(
                type_error_msg.format(f"an object of type {str(type(new_state))}")
            )

        # Check that the new state contains the given columns
        difference = set(self._optimizable_columns).difference(all_columns)
        if difference:
            raise KeyError(
                f"Received columns missing in the provided state: {difference}."
            )
        # An internal current state will save us time for any processing
        # we might need to apply to the current state before subbing in
        # the optimizable variables in the __call__ method.
        self._internal_state = None
        self._actual_state: Vector = pd.concat([self._context_data, new_state], axis=0)

    @property
    def context_data(self) -> Matrix:
        """Context Data getter.

        Returns:
            2d data array
        """
        return self._context_data

    @context_data.setter
    def context_data(self, new_data: Matrix) -> None:
        """Optimizable columns setter.
        Resets internal state.

        Args:
            Context Data
        """
        self._context_data = new_data

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
        params_plus_state = self.substitute_parameters(parameters)
        objectives, params_plus_state = super().__call__(
            params_plus_state, apply_penalties, apply_repairs,
        )
        potentially_repaired_parameters = self._drop_context_values(params_plus_state)
        objectives = self._drop_context_values(objectives)
        return (
            objectives,
            safe_subset(potentially_repaired_parameters, self._optimizable_columns),
        )

    def _drop_context_values(self, data: Vector) -> Vector:
        """
        Since we add context, we'll have to pick rows starting
        from the end of context with stride equals to context size

        Returns:

        """
        context_size = self._context_data.shape[0]
        stride = context_size + 1
        return (
            data[context_size::stride]
            if isinstance(data, np.ndarray)
            else data.iloc[context_size::stride]
        )

    def substitute_parameters(self, parameters: Matrix) -> Matrix:
        """
        Substitute the optimizable parameters into the current state.

        Args:
            parameters: Matrix of parameters to substitute into the current state.

        Returns:
            Matrix of parameters with non-optimizable parameters appended
            in correct order.
        """
        self._update_internal_state_if_shape_changed(parameters)

        # Removes possible collisions with indexes.
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.values

        if self._internal_state is None:
            raise ValueError(f"Internal state wasn't updated")

        # N context rows end at index N-1.
        start = self._context_data.shape[0]
        stride = start + 1
        if isinstance(self._internal_state, np.ndarray):
            np_state_copy = self._internal_state.copy()
            for i in range(self.n_opt_blocks):
                opt_block_start = max(start - i, 0)
                np_state_copy[
                    opt_block_start::stride, self._optimizable_columns,
                ] = parameters
            return np_state_copy
        else:
            pd_state_copy: pd.DataFrame = self._internal_state.copy(deep=True)
            for i in range(self.n_opt_blocks):
                opt_block_start = max(start - i, 0)
                pd_state_copy.loc[
                     opt_block_start::stride, self._optimizable_columns,
                ] = parameters
            return pd_state_copy

    def _update_internal_state_if_shape_changed(self, parameters: Matrix) -> None:
        """
        Updates the internal state
        if the shape of the ``parameters`` matrix has changed.

        Args:
            parameters: Matrix of parameters to evaluate.
        """
        if (
            self._internal_state is None
            or self._internal_state.shape[0] != parameters.shape[0]
        ):
            self._update_internal_state(parameters)

    def _update_internal_state(self, parameters: Matrix) -> None:
        """Update the internal current state variable.

        Args:
            parameters: Matrix of parameters to use to update the internal state.
        """

        n_tile = parameters.shape[0]
        if isinstance(self._actual_state, np.ndarray):
            self._internal_state = np.tile(self._actual_state, (n_tile, 1))
        else:
            self._internal_state = pd.concat(
                [self._actual_state] * n_tile, ignore_index=True
            )
