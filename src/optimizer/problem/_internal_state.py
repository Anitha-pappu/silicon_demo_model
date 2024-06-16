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

from optimizer.types import Matrix, TColumn, Vector


def check_internal_state_needs_update(
    parameters: Matrix,
    internal_state: tp.Optional[Matrix],
    state: Matrix,
) -> bool:
    if internal_state is None:
        return True

    is_same_as_state_type = isinstance(internal_state, type(state))
    is_same_as_state_dim = internal_state.shape[1] == state.shape[1]
    contains_same_columns = (
        internal_state.columns.equals(state)
        if isinstance(internal_state, pd.DataFrame)
        else is_same_as_state_dim  # dims checks is all we can do for an array
    )
    is_same_as_params_length = internal_state.shape[0] == parameters.shape[0]

    return not all([
        is_same_as_state_type,
        is_same_as_state_dim,
        contains_same_columns,
        is_same_as_params_length,
    ])


def create_internal_state(
    parameters: Matrix,
    state: Matrix,
    non_optimizable_columns: tp.List[TColumn],
) -> Matrix:
    state_creator = (
        _create_internal_state_pandas
        if isinstance(state, pd.DataFrame)
        else _create_internal_state_numpy
    )
    return state_creator(
        parameters, state, non_optimizable_columns,
    )


def _create_internal_state_numpy(
    parameters: Matrix,
    state: Matrix,
    non_optimizable_columns: tp.List[TColumn],
) -> Matrix:
    out = np.empty((parameters.shape[0], state.shape[1]), dtype=state.dtype)
    out[:, non_optimizable_columns] = state[:, non_optimizable_columns]  # type: ignore
    return out


def _create_internal_state_pandas(
    parameters: Matrix,
    state: Matrix,
    non_optimizable_columns: tp.List[TColumn],
) -> Matrix:
    index = (
        np.arange(parameters.shape[0])
        if isinstance(parameters, np.ndarray)
        else parameters.index
    )
    if isinstance(state, np.ndarray):
        state = pd.Series(state)

    data = state[non_optimizable_columns].to_dict(orient='list')
    out = pd.DataFrame(data, columns=state.columns, index=index)

    return out
