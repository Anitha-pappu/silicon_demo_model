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

import pandas as pd

_TColumnName = tp.Union[None, str, tp.Tuple[str, ...]]


def check_data(
    data: pd.DataFrame,
    *requested_columns: _TColumnName,
    validate_dataset: bool = True,
    validate_columns: bool = True,
) -> None:
    """
    Validates dataset and columns:
        Dataset:
            - dataset is of `pandas.DataFrame` type
            - dataset is not empty
        Columns:
            - at least one column is requested
            - all requested columns are present in data

    Args:
        data: dataset to validate
        *requested_columns: columns to validate
        validate_dataset: validates dataset if set true
        validate_columns: validates columns if set true

    Raises:
        ValuesError in case any condition is violated
    """
    if validate_dataset:
        _validate_dataset(data)
    if validate_columns:
        _validate_columns(data, *requested_columns)


def _validate_columns(
    data: pd.DataFrame, *requested_columns: _TColumnName,
) -> None:
    all_columns_are_none = all(column is None for column in requested_columns)
    if all_columns_are_none:
        raise ValueError("Please provide variable names for reporting")
    # None is being discarded since None values are coming only
    # when column name is not provided, hence this is not a requested column
    if isinstance(data.columns, pd.MultiIndex):
        not_included_cols = [
            column
            for column in requested_columns
            if column is not None and data.get(column) is None
        ]
        # this is a tricky way of checking single level columns like `("column", "")` in
        # multi-index case;
        # those can be referenced like `df["column"]`
        # but checking `"column" in df.columns` will return `False`
        # since they are stored like ("column", "")
    else:
        not_included_cols = set(requested_columns).difference([None], data.columns)
    if not_included_cols:
        raise ValueError(
            f"The following columns are missing from the dataframe: "
            f"{not_included_cols}",
        )


def _validate_dataset(data: pd.DataFrame) -> None:
    if isinstance(data, pd.Series):
        raise ValueError("Only `pandas.DataFrame` input type is acceptable")
    if data.empty:
        raise ValueError("Nothing to visualize - dataframe is empty")
