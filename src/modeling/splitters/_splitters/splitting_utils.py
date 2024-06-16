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
from copy import copy

import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalarOrArrayConvertible


def verify_split_datetime_inside_series(
    data: pd.Series,
    split_datetime: tp.Union[tp.Any, pd.DatetimeIndex, pd.Timestamp],
) -> None:
    min_datetime = data.min()
    max_datetime = data.max()
    if not min_datetime < split_datetime < max_datetime:  # noqa: WPS508
        raise ValueError(
            f"Provided split_date: '{split_datetime}' lies outside of the "
            f"range of the dataset [{min_datetime},{max_datetime}] ",
        )


def verify_column_inside_dataframe(data: pd.DataFrame, column: str) -> None:
    if column not in data.columns.tolist():
        raise ValueError(f"{column} column is missing")


def verify_frequency_is_valid(freq: str) -> None:
    """
    Verify that a string is a valid argument to pandas frequency-related
    APIs (e.g. '1D', '3W'...).

    Args:
        freq: to validate.

    Returns:
        None - it just passes this frequency to pandas existing function that would
        throw ``ValueError`` if frequency is invalid.
    """
    pd.tseries.frequencies.to_offset(freq)


def convert_datetime(
    timestamp: DatetimeScalarOrArrayConvertible,
    **to_datetime_kwargs: tp.Any,
) -> tp.Union[tp.Any, pd.DatetimeIndex, pd.Timestamp]:
    """Convert datetime value. Calls pd.to_datetime and infers if kwargs not provided.

    Args:
        timestamp: input to pd.to_datetime.
        **to_datetime_kwargs: keyword arguments to pd.to_datetime.

    Returns:
        Output of pd.to_datetime.
    """
    # Pop infer datetime format to avoid collision below.
    to_datetime_kwargs.pop("infer_datetime_format", None)
    infer_datetime_format = "format" not in to_datetime_kwargs
    return pd.to_datetime(
        copy(timestamp),
        infer_datetime_format=infer_datetime_format,
        **to_datetime_kwargs,
    )
