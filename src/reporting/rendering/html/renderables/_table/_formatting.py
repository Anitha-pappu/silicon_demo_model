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

import numpy as np
import pandas as pd
from itables import JavascriptCode, JavascriptFunction
from pandas.api.types import is_numeric_dtype

from reporting.charts.utils import check_data

DEFAULT_TITLE_STYLE = (
    "caption-side: top; "
    "font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; "
    "font-size: 17px; "
    "color: rgb(42, 63, 95); "
    "opacity: 1; "
    "font-weight: normal; "
    "white-space: pre; "
)

_TByColumnPrecision = tp.Dict[tp.Hashable, int]
_TPrecision = tp.Optional[tp.Union[int, _TByColumnPrecision]]

_DEFAULT_PRECISION_KEY = "_default"


logger = logging.getLogger(__name__)


def select_columns(data: pd.DataFrame, columns: tp.List[str]) -> pd.DataFrame:
    if columns:
        check_data(data, *columns)  # todo: use native checeks
        data = data[columns]
    else:
        check_data(data, validate_columns=False)
    return data


def get_formatting_spec(
    data_dtypes: tp.Mapping[str, np.dtype], show_index: bool, precision: _TPrecision,
) -> tp.List[tp.Dict[str, str]]:
    """
    Parses input precision spec

    Args:
        data_dtypes: dataframe's dtypes in dict format
        show_index: true if index is shown
        precision: precision spec

    Returns:
        Parsed itables config
    """
    if precision is None:
        return []

    if isinstance(precision, int):
        by_column_precisions = _by_column_precision_from_int(data_dtypes, precision)
    elif isinstance(precision, dict):
        by_column_precisions = _by_columns_precision_from_dict(data_dtypes, precision)
    else:
        raise ValueError("Precision must be either `int` or `dict[Hashable, int]`")
    _validate_precisions_are_greater_or_equal_than_zero(by_column_precisions)
    precision_spec = _get_precision_spec_for_data_columns(
        data_dtypes,
        by_column_precisions,
        show_index=show_index,
    )
    return precision_spec  # noqa: WPS331  # Naming makes meaning clearer


def _by_column_precision_from_int(
    data_dtypes: tp.Mapping[str, np.dtype],
    precision: int,
):
    """ By-column precision for all numeric types in ``data_dtypes`` assigned as the
    provided ``precision`` value

    Returns a dict where the entries are the names of the numeric columns, and the
    values for all of them is the desired precision ``precision``
    """
    return {
        column: precision
        for column, dtype in data_dtypes.items()
        if is_numeric_dtype(dtype)
    }


def _by_columns_precision_from_dict(
    data_dtypes: tp.Mapping[str, np.dtype],
    precision: tp.Dict[tp.Hashable, int],
):
    precision = precision.copy()
    default_precision = precision.pop(_DEFAULT_PRECISION_KEY, None)
    extra_columns = set(precision).difference(data_dtypes)
    if extra_columns:
        raise ValueError(
            f"For provided precision spec, "
            f"couldn't find following columns in the data: {extra_columns}",
        )
    by_column_precisions = {
        column: precision
        for column, precision in precision.items()
        if column in data_dtypes
    }
    if default_precision is not None:
        by_column_precisions.update(
            {
                column: default_precision
                for column, dtype in data_dtypes.items()
                if column not in precision and is_numeric_dtype(dtype)
            },
        )
    return by_column_precisions


def _get_precision_spec_for_data_columns(
    data_dtypes: tp.Mapping[str, np.dtype],
    by_column_precisions,
    show_index: bool,
):
    """ Get the precision specifications for the columns specified in
    ``by_column_precision``, using their index in the ``data_columns``
    """
    starting_index = 1 if show_index else 0
    data_columns = list(data_dtypes)
    precision_spec = [
        _get_formatting_spec_for_column(
            column_precision, data_columns.index(column) + starting_index,
        )
        for column, column_precision in by_column_precisions.items()
    ]
    return precision_spec  # noqa: WPS331  # Naming makes meaning clearer


# WPS118 in the line below is ok if naming makes meaning clearer
def _validate_precisions_are_greater_or_equal_than_zero(  # noqa: WPS118
    by_column_precisions: tp.Dict[str, int],
) -> None:
    wrong_precisions = [
        column
        for column, column_precision in by_column_precisions.items()
        if column_precision < 0
    ]
    if wrong_precisions:
        raise ValueError(f"Found `precision` < 0 for columns: {wrong_precisions}")


def _get_formatting_spec_for_column(
    precision: int,
    column: int,
    thousands_sep: str = ",",
    precision_delimiter: str = ".",
) -> tp.Dict[str, str]:
    """
    Returns formatting spec {
        "targets": targets_specification,
        "render": js_rendering_code,
    }

    Args:
        precision: number of digits in float representation
        column: column index
    """
    js_formatting_function = (
        "$.fn.dataTable.render.number("
        "'{thousands_sep}', '{precision_delimiter}', {precision})"
    ).format(
        thousands_sep=thousands_sep,
        precision_delimiter=precision_delimiter,
        precision=precision,
    )
    return {
        "targets": column,
        "render": JavascriptCode(js_formatting_function),
    }


def get_bar_coloring_spec(
    df: pd.DataFrame,
    columns_to_draw_bar_for: tp.List[tp.Tuple[int, str]],
    show_index: bool,
) -> tp.List[tp.Dict[str, tp.Any]]:
    if not columns_to_draw_bar_for:
        return []

    if show_index:  # we do that to be able to get index column by name
        # to support older pandas we manually do `df.reset_index(allow_duplicates=True)`
        index_columns = df.index.names
        index_to_reset_into_columns = (
            set(index_columns).difference(df.columns) if index_columns else set()
        )
        df = df.reset_index(level=list(index_to_reset_into_columns))

    return [
        {
            "target": column_index,
            "createdCell": _get_bar_coloring_config_for_column(df[column]),
        }
        for column_index, column in columns_to_draw_bar_for
    ]


def _get_bar_coloring_config_for_column(column_data: pd.Series) -> JavascriptFunction:
    # todo: start coloring based on positive and negative values
    column_data = column_data.abs()  # calculate bar size based on abs values
    column_min = column_data.min()
    column_range = column_data.max() - column_data.min()
    javascript_coloring_fn = """
        function f(td, cellData, rowData, row, col) {
            let cellDataNormalized = (
                Math.abs(cellData) - $COLUMN_MIN
            ) / $COLUMN_RANGE * 100;
            let transparency = Math.min(Math.max(100 - cellDataNormalized - 1, 2), 98);
            $(td).css(
                'background-image',
                `linear-gradient(
                    90deg, transparent ${transparency}%, lightblue ${transparency}%
                )`
            );
            $(td).css('background-size', '98% 88%');
            $(td).css('background-position', 'center center');
            $(td).css('background-repeat', 'no-repeat no-repeat');
        }
        """
    return JavascriptFunction(
        javascript_coloring_fn
        .replace("$COLUMN_MIN", str(column_min))
        .replace("$COLUMN_RANGE", str(column_range)),
    )
