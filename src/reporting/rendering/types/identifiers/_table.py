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

_TColumnName = tp.Union[str, int]
_TSortingConfig = tp.Optional[tp.List[tp.Tuple[_TColumnName, str]]]
_TByColumnPrecision = tp.Dict[tp.Hashable, int]
_TPrecision = tp.Optional[tp.Union[int, _TByColumnPrecision]]


@tp.runtime_checkable
class Table(tp.Protocol):
    """
    Wraps table representation config

    Attributes:
        table: dataframe for chart
        columns: column names of column data to show in the table chart
        precision: sets the number of digits to round float values
        title: title for the chart
        columns_filters_position: position for placing columns filter,
            one of {None, "header", "footer"}; hidden when `None` is passed
        columns_to_color_as_bars: list of column names that will be barcolored
            using the value inside it
        width: the width of the table in layout in percentage
        table_alignment: table alignment, can be one of {"none", "left", "right"}
        sort_by: list for default columns sorting;
            each list element represents a column name and
            its order either "asc" or "desc"
        show_index: shows index column if set True

    Examples::
        Sort by `column_one` ascending and by `column_two` descending
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> Table(df, sort_by=[("column_one", "asc"), ("column_two", "desc")])

        Sort by `column_one` ascending and by `column_two` descending
        >>> df = pd.DataFrame({"index": [1, 3, 2], "column": [6, 4, 5]}).set_index("index")
        >>> Table(df, sort_by=[("index", "asc"), ("column_two", "desc")])

        Set precision for all columns
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> Table(df, precision=2)

        Set precision for specific column and default precision for rest
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> Table(df, precision={"column_one": 1, "_default": 0})
    """  # noqa: E501 – Ok in order to have the code in the string look nice

    table: pd.Series | pd.DataFrame
    columns: list[str] | None
    precision: _TPrecision
    title: str | None
    columns_filters_position: str | None
    columns_to_color_as_bars: list[str] | None
    width: float
    table_alignment: str
    sort_by: _TSortingConfig
    show_index: bool
