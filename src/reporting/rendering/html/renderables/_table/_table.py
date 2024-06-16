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

import pandas as pd
from itables import to_html_datatable as itables_to_html_datatable

from .._base import InteractiveHtmlContentBase  # noqa: WPS436
from ._formatting import (  # noqa: WPS436
    get_bar_coloring_spec,
    get_formatting_spec,
    select_columns,
)
from ._validation import (  # noqa: WPS436
    validate_bars_config,
    validate_sorting_config,
)

DEFAULT_TITLE_STYLE = (
    "caption-side: top; "
    "font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; "
    "font-size: 17px; "
    "color: rgb(42, 63, 95); "
    "opacity: 1; "
    "font-weight: normal; "
    "white-space: pre; "
)

_TColumnName = tp.Union[str, int]
_TSortingConfig = tp.Optional[tp.List[tp.Tuple[_TColumnName, str]]]
_TByColumnPrecision = tp.Dict[tp.Hashable, int]
_TPrecision = tp.Optional[tp.Union[int, _TByColumnPrecision]]

logger = logging.getLogger(__name__)


class InteractiveHtmlRenderableTable(InteractiveHtmlContentBase):
    def __init__(
        self,
        data: tp.Union[pd.Series, pd.DataFrame],
        columns: tp.Optional[tp.List[str]] = None,
        precision: _TPrecision = 4,
        title: tp.Optional[str] = None,
        columns_filters_position: tp.Optional[str] = "footer",
        columns_to_color_as_bars: tp.Optional[tp.List[str]] = None,
        width: float = 50,
        table_alignment: str = "left",
        sort_by: _TSortingConfig = None,
        show_index: bool = True,
    ) -> None:
        """
        Table chart implemented using `itables`.
        This function provides a fancier (compared to plotly) representation of tables
        (with column ordering and filters).

        Args:
            data: dataframe for chart
            columns: column names of column data to show in the table chart
            precision: sets the number of digits to round float values; can be either:
                * int - sets same precision for every numeric column
                * dict - mapping from column name to its precision;
                  "_default" key is optional and confiures the rest of the columns
            title: title for the chart
            columns_filters_position: position for placing columns filter,
                one of {None, "left", "right"}; hidden when `None` is passed
            columns_to_color_as_bars: list of column names that will be barcolored
                using the value inside it
            width: the width of the table in layout in percentage
            table_alignment: table alignment, can be one of {"none", "left", "right"}
            sort_by: list for default columns sorting;
                each list element represents a column name and
                its order either "asc" or "desc".
                Keep index named to be able to sort it, if there is a column
                and index with the same name present and `show_index` is set True,
                then column name will be used for sorting and warning will be thrown
            show_index: shows index column if set True

        Examples::
            Sort by `column_one` ascending and by `column_two` descending
            >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
            >>> plot_table(df, sort_by=[("column_one", "asc"), ("column_two", "desc")])

            Sort by `column_one` ascending and by `column_two` descending
            >>> df = pd.DataFrame({"index": [1, 3, 2], "column": [6, 4, 5]}).set_index("index")
            >>> plot_table(df, sort_by=[("index", "asc"), ("column_two", "desc")])

            Set precision for all columns
            >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
            >>> plot_table(df, precision=2)

            Set precision for specific column and default precision for rest
            >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
            >>> plot_table(df, precision={"column_one": 1, "_default": 0})

        Returns: plotly table chart
        """  # noqa: E501 – Ok to render the line of code nicely
        self.data = data.to_frame() if isinstance(data, pd.Series) else data
        self.columns = columns
        self.precision = precision
        self.title = title
        self.width = width
        self.table_alignment = table_alignment
        self.columns_filters_position = columns_filters_position
        self.columns_to_color_as_bars = validate_bars_config(
            self.data, columns_to_color_as_bars, show_index,
        )
        self.sort_by = validate_sorting_config(self.data, sort_by, show_index)
        self.show_index = show_index

    def to_html(self) -> str:
        return self._repr_html_()

    def _repr_html_(self) -> str:
        prepared_data = select_columns(self.data, self.columns)
        pre_table_tags = (
            (
                f'<caption class="gtitle" style="{DEFAULT_TITLE_STYLE}">'
                f"{self.title}</caption>"
            )
            if self.title is not None
            else ""
        )
        columns_filters_position = (
            self.columns_filters_position
            if self.columns_filters_position is not None
            else False
        )
        return itables_to_html_datatable(
            prepared_data,
            style=f"table-layout:auto;width:{self.width}%;float:{self.table_alignment}",
            classes="hover order-column",
            tags=pre_table_tags,
            column_filters=columns_filters_position,
            order=self.sort_by,
            showIndex=self.show_index,
            columnDefs=[
                {"className": "dt-body-right dt-head-left", "targets": "_all"},
                *get_formatting_spec(
                    prepared_data.dtypes.to_dict(), self.show_index, self.precision,
                ),
                *get_bar_coloring_spec(
                    self.data,
                    self.columns_to_color_as_bars,
                    self.show_index,
                ),
            ],
        )


def plot_table(
    data: pd.DataFrame,
    columns: tp.Optional[tp.List[str]] = None,
    precision: _TPrecision = 4,
    title: tp.Optional[str] = None,
    columns_filters_position: tp.Optional[str] = "footer",
    columns_to_color_as_bars: tp.Optional[tp.List[str]] = None,
    width: float = 50,
    table_alignment: str = "left",
    sort_by: _TSortingConfig = None,
    show_index: bool = True,
) -> InteractiveHtmlRenderableTable:
    """
    Returns table chart rendered using `itables`.
    This function provides a fancier (compared to plotly) representation of tables
    (with column ordering and filters).

    Args:
        data: dataframe for chart
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
        >>> plot_table(df, sort_by=[("column_one", "asc"), ("column_two", "desc")])

        Sort by `column_one` ascending and by `column_two` descending
        >>> df = pd.DataFrame({"index": [1, 3, 2], "column": [6, 4, 5]}).set_index("index")
        >>> plot_table(df, sort_by=[("index", "asc"), ("column_two", "desc")])

        Set precision for all columns
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> plot_table(df, precision=2)

        Set precision for specific column and default precision for rest
        >>> df = pd.DataFrame({"column_one": [1, 3, 2], "column_two": [6, 4, 5]})
        >>> plot_table(df, precision={"column_one": 1, "_default": 0})

    Returns: plotly table chart
    """  # noqa: E501 – Ok in order to have the code in the string look nice

    return InteractiveHtmlRenderableTable(
        data=data,
        columns=columns,
        precision=precision,
        title=title,
        columns_filters_position=columns_filters_position,
        columns_to_color_as_bars=columns_to_color_as_bars,
        width=width,
        table_alignment=table_alignment,
        sort_by=sort_by,
        show_index=show_index,
    )
