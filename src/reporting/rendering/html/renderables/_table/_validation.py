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

import logging
import typing as tp
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)

_TColumnName = tp.Union[str, int]
_TSortingConfig = tp.Optional[tp.List[tp.Tuple[_TColumnName, str]]]

_VALID_SORT_ARGS = frozenset(("asc", "desc"))


def validate_bars_config(
    data: pd.DataFrame,
    columns_to_color_as_bars: tp.Optional[tp.List[str]],
    show_index: bool,
) -> tp.List[tp.Tuple[int, str]]:
    """
    Validates that:
        * all columns are present
        * no duplicated columns are passed and
        * requested columns are numeric

    Notes:
        If there is a column and index with the same name present,
        column name will be used for bar coloring and warning thrown
    """
    if columns_to_color_as_bars is None:
        return []

    if not isinstance(columns_to_color_as_bars, list):
        received_type = type(columns_to_color_as_bars)
        raise ValueError(
            f"`columns_to_color_as_bars` must be of type `None | list`; "
            f"received: {received_type}",
        )

    indexes_names = data.index.names if show_index else []
    index_and_column_names = list(chain(indexes_names, data.columns))

    arg_name = "columns_to_color_as_bars"
    _validate_all_columns_are_known(
        index_and_column_names, columns_to_color_as_bars, arg_name,
    )

    index_and_columns_intersected_with_arg = [
        column for column in index_and_column_names
        if column in set(columns_to_color_as_bars)
    ]  # we keep counts for each column to then search for duplicated
    _validate_no_duplicated_columns(
        index_and_columns_intersected_with_arg, raise_error=False,
    )
    _validate_requested_columns_are_numeric(
        data.reset_index().dtypes.to_dict(),  # add index to dtypes
        index_and_columns_intersected_with_arg,
        arg_name,
    )

    column_to_its_index = {
        column: column_index
        for column_index, column in enumerate(index_and_column_names)
    }
    return [
        (column_to_its_index[column], column)
        for column in columns_to_color_as_bars
    ]


def validate_sorting_config(
    data: pd.DataFrame, sort_by: _TSortingConfig, show_index: bool,
) -> tp.List[tp.Tuple[int, str]]:
    """
    * Maps all column names to itables format i.e. list of (column index, column order)
    * Validates that all columns are present,
      no duplicated columns are passed, all sorting args are either `asc` or `desc`.

    Notes:
        If there is a column and index with the same name present,
        column name will be used for sorting and warning thrown
    """
    if sort_by is None:
        return []

    if not isinstance(sort_by, list):
        received_type = type(sort_by)
        raise ValueError(
            f"`sort_by` must be of type `None | list`; received: {received_type}",
        )

    index_and_column_names = _get_index_and_column_names(data, show_index)
    sort_by_columns = [column for column, _ in sort_by]
    _validate_all_columns_are_known(index_and_column_names, sort_by_columns, "sort_by")
    _validate_no_duplicated_columns(sort_by_columns, raise_error=True)
    _validate_sort_by_order_argument(sort_by)

    index_and_columns_intersected_with_sort = [
        column for column in index_and_column_names if column in set(sort_by_columns)
    ]  # we keep counts for each column to then search for duplicated
    _validate_no_duplicated_columns(
        index_and_columns_intersected_with_sort, raise_error=False,
    )

    column_to_its_index = {
        column: column_index
        for column_index, column in enumerate(index_and_column_names)
    }
    return [(column_to_its_index[column], order) for column, order in sort_by]


def _validate_all_columns_are_known(
    index_and_column_names: tp.List[str],
    sort_by_columns: tp.List[str],
    requesting_arg_name: str,
) -> None:
    unknown_columns = set(sort_by_columns).difference(index_and_column_names)
    if unknown_columns:
        raise ValueError(
            f"Found unknown columns requested in "
            f"`{requesting_arg_name}`: {unknown_columns}",
        )


def _validate_no_duplicated_columns(
    sort_by_columns: tp.List[str], raise_error: bool,
) -> None:
    counter = Counter(sort_by_columns)
    unique_columns = Counter(counter.keys())
    duplicated_columns = list(counter - unique_columns)
    if duplicated_columns:
        err_message = f"Found duplicated columns: {duplicated_columns}"
        if raise_error:
            raise ValueError(err_message)
        else:
            logger.warning(err_message)


def _validate_requested_columns_are_numeric(
    dtypes: tp.Mapping[str, np.dtype], columns: tp.List[str], requesting_arg_name: str,
) -> None:
    non_numeric_columns = [
        column
        for column in columns
        if not is_numeric_dtype(dtypes[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Found non-numeric columns requested by {requesting_arg_name}: "
            f"{non_numeric_columns}",
        )


def _validate_sort_by_order_argument(sort_by: _TSortingConfig) -> None:
    invalid_sorting_args = [
        (columns, sort_arg)
        for columns, sort_arg in sort_by
        if sort_arg not in _VALID_SORT_ARGS
    ]
    if invalid_sorting_args:
        raise ValueError(f"Invalid sorting arg(s) found: {invalid_sorting_args}")


def _get_index_and_column_names(data: pd.DataFrame, show_index: bool):
    """ Gets a list of all the names, starting with the index name and then listing all
    the column names.
    """
    indexes_names = data.index.names if show_index else []
    return list(chain(indexes_names, data.columns))
